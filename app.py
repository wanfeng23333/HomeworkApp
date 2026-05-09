from flask import Flask, request, jsonify
import pymysql
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
app.json.ensure_ascii = False

ALPHA = 0.9
CSV_PATH = "assistments.csv"

def get_conn():
    return pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='123456',
        database='homework_app',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def init_db():
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT PRIMARY KEY AUTO_INCREMENT,
            username VARCHAR(50) NOT NULL UNIQUE,
            password VARCHAR(100) NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_records (
            id INT PRIMARY KEY AUTO_INCREMENT,
            user_id INT NOT NULL,
            subject VARCHAR(100) NOT NULL,
            total_questions INT NOT NULL DEFAULT 0,
            wrong_questions INT NOT NULL DEFAULT 0,
            blank_questions INT NOT NULL DEFAULT 0,
            completion_time FLOAT NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memo (
            id INT PRIMARY KEY AUTO_INCREMENT,
            user_id INT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_features (
            id INT PRIMARY KEY AUTO_INCREMENT,
            user_id INT NOT NULL,
            accuracy FLOAT NOT NULL,
            completion FLOAT NOT NULL,
            efficiency FLOAT NOT NULL,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()

def load_base_centroids():
    df = pd.read_csv(CSV_PATH, encoding="latin1", low_memory=False)
    df = df.head(3000)

    grouped = df.groupby("user_id").agg({
        "correct": "mean",
        "attempt_count": "mean",
        "hint_count": "mean",
        "problem_id": "count"
    }).reset_index()

    grouped.columns = ["user_id", "accuracy", "avg_attempt", "avg_hint", "task_count"]

    scaler = MinMaxScaler()
    grouped["completion"] = scaler.fit_transform(grouped[["task_count"]])
    grouped["efficiency"] = 1 / (1 + grouped["avg_attempt"] + grouped["avg_hint"])

    data = grouped[["accuracy", "completion", "efficiency"]].dropna()

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(data)

    return kmeans.cluster_centers_.tolist()

def get_user_avg_feature(user_id):
    conn = get_conn()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT
                AVG(accuracy) AS accuracy,
                AVG(completion) AS completion,
                AVG(efficiency) AS efficiency
            FROM user_features
            WHERE user_id = %s
        """, (user_id,))
        row = cursor.fetchone()

        if not row or row["accuracy"] is None:
            return [0.5, 0.5, 0.5]

        return [
            float(row["accuracy"]),
            float(row["completion"]),
            float(row["efficiency"])
        ]
    finally:
        cursor.close()
        conn.close()

def update_centroids(base_centroids, user_avg_feature, alpha=ALPHA):
    updated = []
    for c in base_centroids:
        updated.append([
            alpha * c[0] + (1 - alpha) * user_avg_feature[0],
            alpha * c[1] + (1 - alpha) * user_avg_feature[1],
            alpha * c[2] + (1 - alpha) * user_avg_feature[2]
        ])
    return updated

def euclidean_distance(a, b):
    return float(np.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a)))))

def rank_clusters(centroids):
    scored = []
    for i, c in enumerate(centroids):
        score = c[0] * 0.5 + c[1] * 0.3 + c[2] * 0.2
        scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    mapping = {
        scored[0][0]: "优秀",
        scored[1][0]: "中等",
        scored[2][0]: "待提升"
    }
    return mapping

def build_homework_vector(total_questions, wrong_questions, blank_questions, expected_time, actual_time):
    accuracy = (total_questions - wrong_questions - blank_questions) / total_questions
    completion = (total_questions - blank_questions) / total_questions
    efficiency = expected_time / actual_time if actual_time > 0 else 1.0
    efficiency = max(0.0, min(1.0, efficiency))
    return [accuracy, completion, efficiency], accuracy, completion, efficiency

def build_exam_vector(total_score, score, expected_time, actual_time):
    accuracy = score / total_score
    completion = 1.0
    efficiency = expected_time / actual_time if actual_time > 0 else 1.0
    efficiency = max(0.0, min(1.0, efficiency))
    return [accuracy, completion, efficiency], accuracy, completion, efficiency

def save_user_feature(user_id, accuracy, completion, efficiency):
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO user_features (user_id, accuracy, completion, efficiency)
            VALUES (%s, %s, %s, %s)
        """, (user_id, accuracy, completion, efficiency))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def build_suggestion(level, mode):
    if mode == "homework":
        if level == "优秀":
            return "完成情况很好，保持状态即可"
        if level == "中等":
            return "整体稳定，建议减少错题"
        return "基础较弱，建议加强练习"
    else:
        if level == "优秀":
            return "成绩较好，继续保持"
        if level == "中等":
            return "还有提升空间，注意失分点"
        return "需要加强基础训练"

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()

    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()

    if not username or not password:
        return jsonify({
            "code": 400,
            "success": False,
            "message": "用户名或密码不能为空"
        })

    conn = get_conn()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT id, username FROM users WHERE username=%s AND password=%s",
            (username, password)
        )
        user = cursor.fetchone()

        if user:
            return jsonify({
                "code": 200,
                "success": True,
                "message": "登录成功",
                "user_id": user["id"],
                "username": user["username"]
            })
        else:
            return jsonify({
                "code": 401,
                "success": False,
                "message": "用户名或密码错误"
            })
    finally:
        cursor.close()
        conn.close()

@app.route('/api/analyze_manual', methods=['POST'])
def analyze_manual():
    data = request.get_json()
    mode = str(data.get("mode", "")).strip()
    user_id = int(data.get("user_id", 1))

    try:
        if mode == "homework":
            total = int(data.get("total_questions", 0))
            wrong = int(data.get("wrong_questions", 0))
            blank = int(data.get("blank_questions", 0))
            expected = float(data.get("expected_time", 0))
            actual = float(data.get("actual_time", 0))

            user_vector, accuracy, completion, efficiency = build_homework_vector(
                total, wrong, blank, expected, actual
            )

            save_user_feature(user_id, accuracy, completion, efficiency)

            C = load_base_centroids()
            U = get_user_avg_feature(user_id)
            C_new = update_centroids(C, U)

            distances = [euclidean_distance(user_vector, c) for c in C_new]
            cluster_idx = distances.index(min(distances))

            mapping = rank_clusters(C_new)
            level = mapping[cluster_idx]

            if accuracy >= 0.90 and completion >= 0.95 and efficiency >= 0.85:
                level = "优秀"
            elif accuracy >= 0.75 and completion >= 0.80 and efficiency >= 0.60 and level == "待提升":
                level = "中等"

            suggestion = build_suggestion(level, mode)

            return jsonify({
                "success": True,
                "correct_rate": accuracy,
                "error_rate": 1 - accuracy,
                "fine_completion_rate": completion,
                "efficiency": efficiency,
                "level": level,
                "suggestion": suggestion
            })

        else:
            total = float(data.get("total_score", 0))
            score = float(data.get("score", 0))
            expected = float(data.get("expected_time", 0))
            actual = float(data.get("actual_time", 0))

            user_vector, accuracy, completion, efficiency = build_exam_vector(
                total, score, expected, actual
            )

            save_user_feature(user_id, accuracy, completion, efficiency)

            C = load_base_centroids()
            U = get_user_avg_feature(user_id)
            C_new = update_centroids(C, U)

            distances = [euclidean_distance(user_vector, c) for c in C_new]
            cluster_idx = distances.index(min(distances))

            mapping = rank_clusters(C_new)
            level = mapping[cluster_idx]

            if accuracy >= 0.85 and efficiency >= 0.85:
                level = "优秀"
            elif accuracy >= 0.70 and efficiency >= 0.60 and level == "待提升":
                level = "中等"

            suggestion = build_suggestion(level, mode)

            return jsonify({
                "success": True,
                "score_rate": accuracy,
                "loss_rate": 1 - accuracy,
                "efficiency": efficiency,
                "score_text": f"{score}/{total}",
                "level": level,
                "suggestion": suggestion
            })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)