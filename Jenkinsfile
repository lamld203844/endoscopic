pipeline {
  agent {
    docker {
      image 'python'
    }

  }
  stages {
    stage('build') {
      parallel {
        stage('build') {
          steps {
            echo 'Hello, world'
            sh 'python --version'
          }
        }

        stage('build 2') {
          steps {
            echo 'build process 2'
          }
        }

      }
    }

    stage('test') {
      steps {
        echo 'testing....'
      }
    }

    stage('deploy') {
      steps {
        echo 'deploy'
      }
    }

  }
}