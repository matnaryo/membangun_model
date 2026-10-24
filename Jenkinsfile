
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
from pathlib import Path

report = json.loads(
    Path("classification_report.json").read_text()
)

required = ["accuracy", "macro avg", "weighted avg"]
for key in required:
    if key not in report:
        raise SystemExit(f"Metric wajib tidak ada: {key}")

for key in ["accuracy"]:
    value = report[key]
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise SystemExit(f"Metrik tidak valid: {key}={value}")

print("Artefak dan format metrik valid.")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
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

                    echo "Health endpoint berhasil."
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
