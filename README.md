🔥 ¡Bienvenido a mi repositorio de funciones utiles para pytorch! 🔥

👉 Para importar las funciones:

```python
try:
    from pytorch_functions import data_setup, engine, utils, make_predictions
except ImportError:
    # Get the pytorch_functions scripts
    print("[INFO] Downloading.")
    !git clone https://github.com/Andresmup/pytorch_functions
    from pytorch_functions import data_setup, engine, utils, make_predictions
```

🐍 Puedes ejecutarlas desde la linea de comandos pasando los parametros 🐍
```
!python pytorch_functions/train_tinyvgg.py --model model --batch_size 32 --lr 0.001 --num_epochs 5 --name_saving modelo_guardado
```

✅ Si estas interesado en conocer el uso que le doy a estas funciones, necesitas ayuda, no dudes contactarme y/o visitar mis otros repositorios. 

💬 Gracias por visitar. Escribeme si tienes ideas para mejorar las funciones. 💬

👨‍💻 Andrés Muñoz Pampillón 
