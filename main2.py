import pandas as pd
import numpy as np
import re
from fractions import Fraction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

# Function to convert time strings to minutes
def time_to_minutes(time_str):
    if pd.isnull(time_str):
        return None
    time_str = time_str.lower()
    hours = re.search(r'(\d+)\s*hrs?', time_str)
    minutes = re.search(r'(\d+)\s*mins?', time_str)
    total_minutes = 0
    if hours:
        total_minutes += int(hours.group(1)) * 60
    if minutes:
        total_minutes += int(minutes.group(1))
    return total_minutes if total_minutes > 0 else None


# Function to parse yield values, including fractions
def parse_yield(value):
    try:
        return float(sum(Fraction(part) for part in value.split()))
    except (ValueError, TypeError, AttributeError):
        return np.nan


# Load the dataset
data = pd.read_csv('recipes.csv')

# Preprocess time columns
data['prep_time'] = data['prep_time'].apply(time_to_minutes)
data['cook_time'] = data['cook_time'].apply(time_to_minutes)
data['total_time'] = data['total_time'].apply(time_to_minutes)

# Handle missing values in time columns 
def fill_missing_times(row):
    if pd.isnull(row['total_time']):
        row['total_time'] = row['prep_time'] + row['cook_time']
    if pd.isnull(row['cook_time']):
        row['cook_time'] = row['total_time'] - row['prep_time']
    if pd.isnull(row['prep_time']):
        row['prep_time'] = row['total_time'] - row['cook_time']
    
    # Ensure no negative times
    row['total_time'] = max(row['total_time'], 0)
    row['cook_time'] = max(row['cook_time'], 0)
    row['prep_time'] = max(row['prep_time'], 0)
    
    return row

data = data.apply(fill_missing_times, axis=1)

# Process the 'yield' column
data['yield'] = data['yield'].apply(parse_yield)
data['yield'].fillna(data['yield'].median(), inplace=True)

# Standardize numerical columns
numerical_cols = ['prep_time', 'cook_time', 'total_time', 'yield']
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Encode categorical columns
label_encoder = LabelEncoder()
if 'cuisine_path' in data.columns:
    data['cuisine_path'] = label_encoder.fit_transform(data['cuisine_path'])

# Feature engineering
data['total_time_per_serving'] = data['total_time'] / data['yield']
data['ingredient_count'] = data['ingredients'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
data['prep_cook_interaction'] = data['prep_time'] * data['cook_time']

# Train-test split
X = data[['prep_time', 'cook_time', 'total_time', 'yield', 'total_time_per_serving', 'ingredient_count', 'prep_cook_interaction']]
y = data['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)

# Evaluate model
y_pred_rf = rf_regressor.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Mean Squared Error: {mse_rf}")
print(f"Random Forest R² Score: {r2_rf}")


# Tkinter Application
class CookbookApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cookbook")
        self.root.geometry("800x600")
        self.root.configure(bg="#F4F1DE")
        self.show_welcome_screen()

    def show_welcome_screen(self):
        self.clear_window()

        # Display image
        try:
            welcome_img = Image.open("image.png")
            welcome_img = welcome_img.resize((800, 450), Image.Resampling.LANCZOS)
            self.welcome_photo = ImageTk.PhotoImage(welcome_img)
            img_label = tk.Label(self.root, image=self.welcome_photo, bg="#F4F1DE")
            img_label.pack(pady=20)
        except Exception as e:
            tk.Label(
                self.root,
                text="Image could not be loaded.",
                font=("Helvetica", 14),
                bg="#F4F1DE",
                fg="red"
            ).pack(pady=20)

        # Proceed Button
        proceed_button = tk.Button(
            self.root,
            text="Proceed",
            command=self.create_ingredients_page,
            bg="#81B29A",
            fg="white",
            font=("Helvetica", 16, "bold")
        )
        proceed_button.pack(side=tk.BOTTOM, pady=30)

    def create_ingredients_page(self):
        self.clear_window()

        ingredients_label = tk.Label(
            self.root,
            text="Enter ingredients (comma-separated):",
            font=("Helvetica", 14),
            bg="#F4F1DE"
        )
        ingredients_label.pack(pady=10)

        self.ingredients_entry = tk.Entry(self.root, font=("Helvetica", 14), width=40)
        self.ingredients_entry.pack(pady=10)

        search_button = tk.Button(
            self.root,
            text="Search Recipes",
            command=self.search_recipes,
            bg="#81B29A",
            fg="white",
            font=("Helvetica", 14, "bold")
        )
        search_button.pack(pady=10)

        self.results_frame = tk.Frame(self.root, bg="#F4F1DE")
        self.results_frame.pack(fill="both", expand=True)

    def search_recipes(self):
        query = self.ingredients_entry.get().lower()
        if not query:
            messagebox.showerror("Input Error", "Please enter at least one ingredient.")
            return

        matching_recipes = data[data['ingredients'].str.contains(query, case=False, na=False)]

        for widget in self.results_frame.winfo_children():
            widget.destroy()

        if matching_recipes.empty:
            tk.Label(
                self.results_frame,
                text="No recipes found.",
                font=("Helvetica", 14),
                bg="#F4F1DE"
            ).pack(pady=20)
        else:
            for _, recipe in matching_recipes.iterrows():
                recipe_button = tk.Button(
                    self.results_frame,
                    text=recipe['recipe_name'],
                    command=lambda name=recipe['recipe_name']: self.show_recipe_details(name),
                    font=("Helvetica", 12),
                    bg="#E07A5F",
                    fg="white",
                    wraplength=500
                )
                recipe_button.pack(pady=5, padx=10, anchor="w", fill="x")

    def show_recipe_details(self, recipe_name):
        self.clear_window()

        # Creating a scrollable frame
        canvas = tk.Canvas(self.root, bg="#F4F1DE")
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        scrollable_frame = tk.Frame(canvas, bg="#F4F1DE")
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Adding recipe details
        recipe = data[data['recipe_name'] == recipe_name].iloc[0]

        title_label = tk.Label(
            scrollable_frame,
            text=recipe_name,
            font=("Helvetica", 18, "bold"),
            bg="#F4F1DE",
            wraplength=700
        )
        title_label.pack(pady=20)

        details_text = f"""
        Ingredients: {recipe['ingredients']}
        
        Prep Time: {recipe['prep_time']}
        Cook Time: {recipe['cook_time']}
        Total Time: {recipe['total_time']}
        Yield: {recipe['yield']}
        
        Directions: {recipe['directions']}
        
        Nutrition: {recipe['nutrition']}
        """
        details_label = tk.Label(
            scrollable_frame,
            text=details_text,
            font=("Helvetica", 12),
            bg="#F4F1DE",
            justify="left",
            wraplength=700
        )
        details_label.pack(pady=10)

        back_button = tk.Button(
            scrollable_frame,
            text="Back",
            command=self.create_ingredients_page,
            bg="#81B29A",
            fg="white",
            font=("Helvetica", 12, "bold")
        )
        back_button.pack(pady=20)

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()


# Run the Application
root = tk.Tk()
app = CookbookApp(root)
root.mainloop()