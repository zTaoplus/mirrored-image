FROM zjuici/mirror.huggingface.text-generation-inference:1.1.1-transformers-4.34

RUN pip install -U pygtrie -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir

COPY tokens.py /opt/conda/lib/python3.9/site-packages/text_generation_server/utils
COPY flash_causal_lm.py /opt/conda/lib/python3.9/site-packages/text_generation_server/models
COPY model.py /opt/conda/lib/python3.9/site-packages/text_generation_server/models