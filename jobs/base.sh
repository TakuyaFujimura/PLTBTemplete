version=$1

cd ../
source venv/bin/activate

export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")

python train.py experiments="${version}" 'version='${version}''
