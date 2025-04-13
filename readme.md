# Title
This is machine learning learning notes

## Env
建立虛擬環境:
```
py -m venv mlenv
```

啟動虛擬環境:
```
mlenv\Scripts\activate
```

停用虛擬環境:
```
deactivate
```

## Python command
輸出已安裝套件到requirements:
```
pip freeze > requirements.txt
```

安裝requirements裡的套件(要先啟動虛擬環境，套件才會安裝在正確的環境裡)
```
pip install -r requirements.txt
```