
pipeline {
    agent any

    environment {
        IMAGE_NAME = 'matnaryo/loan_default_tuning_lr'
        CONTAINER_NAME = 'Loan-Default-Tuning-LR'
        TEST_CONTAINER = 'loan-default-ci-test'
        MODEL_DIR = 'loan_default_tuning_lr/model_dir/model'
        HOST_PORT = '5005'
        TEST_PORT = '5006'
    }

    stages {
        stage('Checkout & Validate') {
            steps {
                checkout scm

                sh '''
                    set -eu

                    test -f "$MODEL_DIR/MLmodel"
                    test -f "$MODEL_DIR/model.pkl"
                    test -f "$MODEL_DIR/requirements.txt"
                    test -f loan_default_tuning_lr/Dockerfile
                    test -f classification_report.json

                    python3 - <<'PY'
import json
import sys

report_path = "classification_report.json"

MIN_ACCURACY = 0.65
MIN_MACRO_F1 = 0.55
MIN_RECALL_CLASS_1 = 0.65

try:
    with open(report_path, encoding="utf-8") as file:
        report = json.load(file)

    accuracy = float(report["accuracy"])
    macro_f1 = float(report["macro avg"]["f1-score"])
    recall_class_1 = float(report["1"]["recall"])

except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f"GAGAL: laporan metrik tidak valid: {exc}")
    sys.exit(1)

metrics = {
    "Accuracy": (accuracy, MIN_ACCURACY),
    "Macro F1": (macro_f1, MIN_MACRO_F1),
    "Recall kelas 1": (
        recall_class_1,
        MIN_RECALL_CLASS_1,
    ),
}

for name, (value, minimum) in metrics.items():
    print(f"{name}: {value:.4f} (minimum {minimum:.4f})")

    if not 0 <= value <= 1:
        print(f"GAGAL: {name} di luar rentang 0–1.")
        sys.exit(1)

    if value < minimum:
        print(f"GAGAL: {name} tidak memenuhi ambang minimum.")
        sys.exit(1)

print("LULUS: semua metrik memenuhi ambang minimum.")
PY
        '''
    }
}


        stage('Build Candidate Image') {
            steps {
                sh '''
                    set -eu
                    docker build \
                      -t "${IMAGE_NAME}:candidate-${BUILD_NUMBER}" \
                      -f loan_default_tuning_lr/Dockerfile \
                      loan_default_tuning_lr
                '''
            }
        }


        stage('Test Candidate') {
            steps {
                sh '''
                    set -eu

                    docker rm -f "$TEST_CONTAINER" >/dev/null 2>&1 || true

                    cleanup() {
                        docker rm -f "$TEST_CONTAINER" >/dev/null 2>&1 || true
                    }
                    trap cleanup EXIT

                    docker run -d \
                      --name "$TEST_CONTAINER" \
                      -p "127.0.0.1:${TEST_PORT}:8080" \
                      "${IMAGE_NAME}:candidate-${BUILD_NUMBER}"

                    ready=0
                    for i in $(seq 1 30); do
                        if curl -fsS \
                          "http://127.0.0.1:${TEST_PORT}/ping" >/dev/null; then
                            ready=1
                            break
                        fi
                        sleep 2
                    done

                    if [ "$ready" -ne 1 ]; then
                        docker logs "$TEST_CONTAINER"
                        echo "Model serving tidak sehat."
                        exit 1
                    fi

                    echo "Health check berhasil."

                    python3 - <<'PY'
import json
import os
import urllib.request
import urllib.error

columns = [
    "Age", "Income", "LoanAmount", "CreditScore",
    "MonthsEmployed", "NumCreditLines", "InterestRate",
    "LoanTerm", "DTIRatio",
    "Education_Bachelor's", "Education_High School",
    "Education_Master's", "Education_PhD",
    "EmploymentType_Full-time", "EmploymentType_Part-time",
    "EmploymentType_Self-employed", "EmploymentType_Unemployed",
    "MaritalStatus_Divorced", "MaritalStatus_Married",
    "MaritalStatus_Single",
    "HasMortgage_No", "HasMortgage_Yes",
    "HasDependents_No", "HasDependents_Yes",
    "LoanPurpose_Auto", "LoanPurpose_Business",
    "LoanPurpose_Education", "LoanPurpose_Home",
    "LoanPurpose_Other",
    "HasCoSigner_No", "HasCoSigner_Yes",
]

row = [
    35, 5000, 15000, 700, 36, 4, 8.5, 36, 0.25,
    1, 0, 0, 0,       # Education: Bachelor's
    1, 0, 0, 0,       # EmploymentType: Full-time
    0, 1, 0,          # MaritalStatus: Married
    0, 1,             # HasMortgage: Yes
    0, 1,             # HasDependents: Yes
    0, 0, 0, 1, 0,    # LoanPurpose: Home
    1, 0,             # HasCoSigner: No
]

assert len(columns) == 31, f"Jumlah kolom salah: {len(columns)}"
assert len(row) == len(columns), "Jumlah nilai tidak sesuai kolom"

payload = {
    "dataframe_split": {
        "columns": columns,
        "data": [row],
    }
}

url = f"http://127.0.0.1:{os.environ['TEST_PORT']}/invocations"
request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        status = response.status
        body = response.read().decode("utf-8")
except urllib.error.HTTPError as exc:
    print(f"Prediksi gagal: HTTP {exc.code}")
    print(exc.read().decode("utf-8", errors="replace"))
    raise SystemExit(1)
except Exception as exc:
    print(f"Request prediksi gagal: {exc}")
    raise SystemExit(1)

if status != 200:
    print(f"Prediksi gagal: HTTP {status}")
    print(body)
    raise SystemExit(1)

try:
    result = json.loads(body)
except json.JSONDecodeError:
    print("Respons server bukan JSON yang valid.")
    print(body)
    raise SystemExit(1)

predictions = result.get("predictions")
if not isinstance(predictions, list) or len(predictions) != 1:
    print("Respons tidak berisi tepat satu prediksi.")
    print(result)
    raise SystemExit(1)

print(f"Tes prediksi berhasil. HTTP {status}")
print(f"Hasil prediksi: {predictions[0]}")
PY
                '''
            }
        }


        stage('Deploy') {
            steps {
                input message: 'Image kandidat sehat. Lanjutkan deployment ke port 5005?', ok: 'Deploy'

                sh '''
                    set -eu

                    OLD_IMAGE=$(docker inspect \
                      --format='{{.Image}}' "$CONTAINER_NAME")

                    NEW_IMAGE="${IMAGE_NAME}:candidate-${BUILD_NUMBER}"
                    BACKUP_IMAGE="${IMAGE_NAME}:rollback-${BUILD_NUMBER}"

                    docker tag "$OLD_IMAGE" "$BACKUP_IMAGE"
                    docker tag "$NEW_IMAGE" "${IMAGE_NAME}:latest"

                    docker rm -f "${CONTAINER_NAME}-previous" \
                      >/dev/null 2>&1 || true

                    docker stop "$CONTAINER_NAME"
                    docker rename "$CONTAINER_NAME" "${CONTAINER_NAME}-previous"

                    if ! docker run -d \
                      --name "$CONTAINER_NAME" \
                      --restart unless-stopped \
                      -p "${HOST_PORT}:8080" \
                      "${IMAGE_NAME}:latest"; then

                        docker rm -f "$CONTAINER_NAME" \
                          >/dev/null 2>&1 || true
                        docker rename "${CONTAINER_NAME}-previous" "$CONTAINER_NAME"
                        docker start "$CONTAINER_NAME"
                        exit 1
                    fi

                    healthy=0
                    for i in $(seq 1 30); do
                        if curl -fsS \
                          "http://127.0.0.1:${HOST_PORT}/ping" >/dev/null; then
                            healthy=1
                            break
                        fi
                        sleep 2
                    done

                    if [ "$healthy" -ne 1 ]; then
                        docker logs "$CONTAINER_NAME"
                        docker rm -f "$CONTAINER_NAME" \
                          >/dev/null 2>&1 || true
                        docker rename "${CONTAINER_NAME}-previous" "$CONTAINER_NAME"
                        docker start "$CONTAINER_NAME"
                        exit 1
                    fi

                    echo "Deployment berhasil. Container lama dipertahankan sebagai cadangan."
                '''
            }
        }
    }

    post {
        always {
            sh '''
                docker rm -f loan-default-ci-test >/dev/null 2>&1 || true
            '''
        }
        success {
            echo 'Pipeline berhasil.'
        }
        failure {
            echo 'Pipeline gagal. Periksa log stage yang gagal.'
        }
    }
}
