# Aneurysm Pressure Prediction using 3D Geometry

This project predicts pressure, velocity-gradient and wall shear stress in cerebral vasculature using machine learning models trained on geometrical and physical features extracted from DICOM-derived datasets.

---

## Project Structure

```
aneurysm-pressure-prediction/
├── data/
│   └── APV2014A2-1-04500_diastole.csv
├── models/
├── results/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── logger.py
│   ├── main.py
│   └── __init__.py
├── notebooks/
│   └── exploration.py
├── assets/
│   └── brain-vascular-geometry.png
├── docs/
│   └── index.html
├── .gitignore
├── README.md
└── requirements.txt
```


---

## 🚀 How to Run

### Step 1: Install dependencies

pip install -r requirements.txt


### Step 2: Train or Predict

python src/main.py


The model will train on the provided vascular geometry and pressure datasets.  
You can modify `main.py` to input new coordinates and predict pressure using the trained models.

---

## Model Details

This pipeline supports several machine learning models, including:

- Random Forest Regressor
- Support Vector Regression (SVR)
- Linear Regression
- Convolution Neural Network

It uses 3D coordinate-based vascular geometry and associated pressure/velocity metrics for training.  
Once trained, the model can return pressure predictions for new geometrical inputs. Similarly it can predict velocity-gradient and wall shear stress based on the geometry.

---

## Data Notes

- All input CSVs should be placed in the `data/` directory.
- Prediction results will be saved in the `results/` directory.
- Trained model files are saved to or loaded from the `models/` directory.


---

## Tech Stack

- Python
- Mimics
- Blender
- Ansys Fluent
- Scikit-learn
- NumPy
- Pandas
- Matplotlib / Seaborn
- Paraview
- Git & GitHub

---

## Testing

To test with new coordinate data:

- Update the inference logic in `main.py`
- Ensure the model is either trained or loaded

---

## Contact

- [LinkedIn](https://www.linkedin.com/in/pushyan-jhaveri-75b52813a/)
- [Website](https://pushyanjhaveri.github.io/)

