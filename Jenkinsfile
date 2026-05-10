pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'heart-disease-api'
        DOCKER_TAG = "${BUILD_NUMBER}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python Environment') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '''
                    . venv/bin/activate
                    flake8 src/ tests/ --max-line-length=120 --exclude=__pycache__
                '''
            }
        }

        stage('Prepare Data') {
            steps {
                sh '''
                    . venv/bin/activate
                    python data/download_data.py
                    python -c "from src.data_prep.prepare import run; run()"
                '''
            }
        }

        stage('Unit Tests') {
            steps {
                sh '''
                    . venv/bin/activate
                    python -m pytest tests/ -v --tb=short --junitxml=test-results.xml
                '''
            }
            post {
                always {
                    junit 'test-results.xml'
                }
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                    . venv/bin/activate
                    python -c "from src.training.train import run; run()"
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh "docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} -t ${DOCKER_IMAGE}:latest ."
            }
        }

        stage('Push Docker Image') {
            when {
                branch 'main'
            }
            steps {
                sh """
                    docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} localhost:5000/${DOCKER_IMAGE}:${DOCKER_TAG}
                    docker push localhost:5000/${DOCKER_IMAGE}:${DOCKER_TAG}
                """
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'model/artifacts/**', fingerprint: true
            archiveArtifacts artifacts: 'test-results.xml', fingerprint: true
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs above for details.'
        }
    }
}
