# Question Answering - BERT

Requiere descargar el dataset y colocarlo en el directorio raíz del proyecto

https://huggingface.co/mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es

```
# Instalar modulo para entorno virtual
pip install virtualenv

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
venv\Scripts\activate (windows)
source venv/bin/activate (linux)

# Instalar modulos del proyecto
pip install -r requirements.txt

Se debe ejecutar ambos programas:

python ml_api.py

streamlit run ml_ui.py
```
