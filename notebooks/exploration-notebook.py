{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Velocity Prediction Data Exploration\n",
    "\n",
    "This notebook explores the datasets used for velocity prediction based on spatial coordinates (x, y, z)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Set style for plotting\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "sns.set_context(\"notebook\", font_scale=1.2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load Configuration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import configuration from parent directory\n",
    "import sys\n",
    "sys.path.append('..')\n",
    "from config import TRAINING_FILES, TEST_FILES, FEATURE_COLUMNS, TARGET_COLUMN, DATA_DIR"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Loading"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load a sample CSV file\n",
    "sample_file = TRAINING_FILES[0]\n",
    "file_path = os.path.join(DATA_DIR, sample_file)\n",
    "\n",
    "try:\n",
    "    df = pd.read_csv(file_path)\n",
    "    print(f\"Successfully loaded {sample_file}\")\n",
    "    print(f\"Shape: {df.shape}\")\n",
    "except Exception as e:\n",
    "    print(f\"Error loading {sample_file}: {e}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Overview"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display the first few rows\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Summary statistics\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check for missing values\n",
    "print(\"Missing values per column:\")\n",
    "df.isna().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Feature Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Extract features and target\n",
    "features = df.iloc[:, FEATURE_COLUMNS]\n",
    "target = df.iloc[:, TARGET_COLUMN]\n",
    "\n",
    "# Rename columns for clarity\n",
    "features.columns = ['X', 'Y', 'Z']\n",
    "\n",
    "# Show the features\n",
    "features.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a dataframe with features and target for analysis\n",
    "analysis_df = pd.concat([features, target.rename('Velocity')], axis=1)\n",
    "analysis_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Distribution of Target Variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(analysis_df['Velocity'], kde=True)\n",
    "plt.title('Distribution of Velocity Values')\n",
    "plt.xlabel('Velocity')\n",
    "plt.ylabel('Frequency')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Feature-Target Relationships"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scatter plots for each feature vs. target\n",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "\n",
    "for i, feature in enumerate(['X', 'Y', 'Z']):\n",
    "    sns.scatterplot(x=feature, y='Velocity', data=analysis_df, alpha=0.5, ax=axes[i])\n",
    "    axes[i].set_title(f'{feature} vs. Velocity')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Correlation Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation matrix\n",
    "corr_matrix = analysis_df.corr()\n",
    "\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\n",
    "plt.title('Correlation Matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3D Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3D scatter plot with velocity as color\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "\n",
    "fig = plt.figure(figsize=(12, 10))\n",
    "ax = fig.add_subplot(111, projection='3d')\n",
    "\n",
    "# Sample a subset if the dataset is large\n",
    "sample_size = min(1000, len(analysis_df))\n",
    "sample_df = analysis_df.sample(sample_size, random_state=42)\n",
    "\n",
    "scatter = ax.scatter(\n",
    "    sample_df['X'], \n",
    "    sample_df['Y'], \n",
    "    sample_df['Z'],\n",
    "    c=sample_df['Velocity'],\n",
    "    cmap='viridis',\n",
    "    alpha=0.7\n",
    ")\n",
    "\n",
    "ax.set_xlabel('X')\n",
    "ax.set_ylabel('Y')\n",
    "ax.set_zlabel('Z')\n",
    "plt.colorbar(scatter, ax=ax, label='Velocity')\n",
    "plt.title('3D Spatial Distribution with Velocity as Color')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Exploration Across Multiple Files"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load multiple files and analyze velocity distributions\n",
    "num_samples = min(5, len(TRAINING_FILES))  # Analyze first 5 files\n",
    "\n",
    "plt.figure(figsize=(12, 6))\n",
    "\n",
    "for i, file in enumerate(TRAINING_FILES[:num_samples]):\n",
    "    try:\n",
    "        file_path = os.path.join(DATA_DIR, file)\n",
    "        sample_df = pd.read_csv(file_path)\n",
    "        velocity = sample_df.iloc[:, TARGET_COLUMN]\n",
    "        \n",
    "        sns.kdeplot(velocity