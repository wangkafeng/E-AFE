python=3.6.8
pip install  numpy scipy pandas sklearn xlwt xlutils tensorflow==1.13.1 keras==2.0.0 weka rpyc javabridge python-weka-wrapper3 auto-sklearn pytorch

python Main_weka_minhash.py
python Main_sklearn_minhash.py
python Main_sklearn_cnn.py

The source code refers to the implementation of NFS （Neural feature search: A neural architecture for automated feature engineering).
MinHash function library refers to the implementation of Wei Wu's MinHash libray (A review for weighted minhash algorithms).


This code is an implementation of the E-AFE method, and the paper information is as follows.
Toward Efficient Automated Feature Engineering
2023 IEEE 39th International Conference on Data Engineering (ICDE)
https://ieeexplore.ieee.org/document/10184603
