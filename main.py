# Import required libraries
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import json
from scipy.optimize import minimize
from config import filter_words, foods_to_eat, chosen_words_language, label_language, label

# Load the new data
with open('json/dri_mapping.json', 'r', encoding='utf-8') as f:
    mapping_data = json.load(f)

with open('json/nutrients_database.json', 'r', encoding='utf-8') as f:
    nutrients_data = json.load(f)


# Adjusting the filter for nutrients_data
if filter_words:
    foods_to_eat = [food.lower() for food in foods_to_eat]
    nutrients_data = {k: v for k, v in nutrients_data.items(
    ) if f'short_name_{chosen_words_language}' in v and v[f'short_name_{chosen_words_language}'].lower() in foods_to_eat}


def f_to_optimize(x, nutrient_matrix, dri):
    y = nutrient_matrix @ x
    return np.sum(((y - dri) / dri) ** 2)


def find_optimal_food_amount(nutrient_df, nutrient_constraints_df):
    nutrient_matrix = nutrient_df.values  # Convert DataFrame to NumPy array
    num_nutrients, num_foods = nutrient_matrix.shape

    # Extract the DRIs
    dri = nutrient_constraints_df['dri'].values

    # Set the bounds for the decision variables
    bounds = [(0, None) for _ in range(num_foods)]

    # Define the starting point for the optimization
    start_point = np.zeros(num_foods)

    # Solve the optimization problem
    result = minimize(f_to_optimize, start_point, args=(
        nutrient_matrix, dri), bounds=bounds)

    return result


# Conversion of the data to required DataFrame format
nutrient_constraints_df_new = pd.DataFrame(mapping_data).T[['dri', 'ul']]
selected_nutrients = nutrient_constraints_df_new.index.tolist()

foods_data = {}
for food_id, food_data in nutrients_data.items():
    food_name = food_data[f"short_name_{label_language}"]
    nutrients = {n["nutrient_name"]: n["food_nutrient_amount"]
                 for n in food_data["nutrients"]}
    foods_data[food_name] = nutrients

# After creating foods_data and before converting it to DataFrame
nutrient_df_new = pd.DataFrame(foods_data).T.fillna(0)

# Filter only the common nutrients between nutrient_df_new and selected_nutrients
common_nutrients = list(set(nutrient_df_new.columns) & set(selected_nutrients))
nutrient_df_new = nutrient_df_new[common_nutrients]

nutrient_df_new = nutrient_df_new.T


# Extracting names and their corresponding dri and ul values
df_data = {k: (v["dri"], v["ul"]) for k, v in mapping_data.items()}

# Converting into DataFrame
nutrient_constraints_df_new = pd.DataFrame(df_data, index=["dri", "ul"]).T
# nutrient_constraints_df_new

# Reindex nutrient_df_new based on the index of nutrient_constraints_df_new
nutrient_df_new = nutrient_df_new.reindex(nutrient_constraints_df_new.index)

# Fill NaN values with zeros
nutrient_df_new.fillna(0, inplace=True)

# Find the optimal food amounts
result = find_optimal_food_amount(nutrient_df_new, nutrient_constraints_df_new)


# Calculate the nutrient contributions of each food
nutrient_contributions = nutrient_df_new * result.x

# Normalize the nutrient contributions and DRI by UL
normalized_nutrient_contributions = nutrient_contributions.div(
    nutrient_constraints_df_new['ul'], axis=0)
normalized_dri = nutrient_constraints_df_new['dri'] / \
    nutrient_constraints_df_new['ul']

# Create the stacked bar chart
fig = go.Figure()

bottom_rect_width = 1
bottom_rect_width /= 2
# Add rectangles for the background
for i, (nutrient, norm_dri) in enumerate(zip(selected_nutrients, normalized_dri)):
    fig.add_shape(
        type='rect',
        x0=0,
        x1=norm_dri,
        y0=i - bottom_rect_width,
        y1=i + bottom_rect_width,
        xref='x',
        yref='y',
        fillcolor='red',
        opacity=0.2,
        layer='below',
        line_width=0
    )
    fig.add_shape(
        type='rect',
        x0=norm_dri,
        x1=1,
        y0=i - bottom_rect_width,
        y1=i + bottom_rect_width,
        xref='x',
        yref='y',
        fillcolor='green',
        opacity=0.2,
        layer='below',
        line_width=0
    )
    fig.add_shape(
        type='rect',
        x0=1,
        x1=1.2,
        y0=i - bottom_rect_width,
        y1=i + bottom_rect_width,
        xref='x',
        yref='y',
        fillcolor='red',
        opacity=0.2,
        layer='below',
        line_width=0
    )

food_translated = {data[f'short_name_{label_language}']: data[f'short_name_{label_language}']
                   for food_id, data in nutrients_data.items()}

selected_nutrients_ru = [mapping_data[nutrient]
                         [f'name_{label_language}'] for nutrient in selected_nutrients]

# Add bar traces for each food
for food, data in normalized_nutrient_contributions.items():
    grams = round(data.iloc[1] * 1000, 2)
    fig.add_trace(go.Bar(
        x=data,
        y=selected_nutrients_ru,
        name=str(grams) + ' г, ' + food_translated[food],
        marker=dict(line=dict(width=1)),
        orientation='h',
        width=0.5,  # Adjust bar width here
    ))

# Customize the chart appearance
fig.update_layout(
    barmode='stack',
    # title="Оптимальное количество пищи и содержание питательных веществ (нормализовано)",
    title='',
    xaxis_title=label,
    xaxis=dict(range=[0, 1.2]),
    legend=dict(orientation="h", yanchor="bottom",
                y=1.02, xanchor="right", x=1),
    margin=dict(t=100, b=100),
)

desired_row_height = 50
num_nutrients = len(selected_nutrients_ru)
# 200 pixels for padding, titles, legend, etc.
fig_height = desired_row_height * num_nutrients + 200
fig.update_layout(height=fig_height)


# Adjust the spacing between bars
fig.update_yaxes(automargin=True, tickson="boundaries",
                 ticklen=10, tickwidth=2, tickcolor="white")

# Show the chart
fig.show()
