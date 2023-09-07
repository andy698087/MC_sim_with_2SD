import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns  
from adjustText import adjust_text

# Create a DataFrame with your data, including N1=50
df = pd.read_csv('From_text_GPM_MC_nMonte_100000_N_Higgins12MoMNo_SE.csv')

col_new_names = []
for col in df.columns:
    col_new_names.append(col.strip())

df.columns = col_new_names
df = df[df['N'] != 2]
df = df[df['MethodOfMoments'] != 'Higgins2']
# df = df[['MethodOfMoments','N','CV', 'ln ratio SE include zero']]

methods = [df["MethodOfMoments"].unique()[i] for i in range(len(df["MethodOfMoments"].unique()))]

# Define markers for each MethodOfMoments
# marker_styles = {methods[0]: 'o', methods[1]: 's', methods[2]: '^', methods[3]: 'D'}
marker_styles = {methods[0]: 'o', methods[1]: '^', methods[2]: 'D'}
colors = sns.color_palette("husl", n_colors=len(df["MethodOfMoments"].unique()) * len(df["N"].unique()))



# Create a grouped line chart
plt.figure(figsize=(15, 9))
for (method, n1), data in df.groupby(["MethodOfMoments", "N"]):
    color = colors.pop() # Get the next color from the palette
    marker_style = marker_styles.get(method, 'o')
    # Add the legend with standard error
    se_value = data[data["MethodOfMoments"] == method]["intervals_include_zero_SE"].iloc[0]
    plt.plot(data["CV"], data["intervals_include_zero_mean"], marker=marker_style, label=f"{method}, N={n1:.0f}, SE={se_value:.6f}", color=color)
    
    # Add text with the mean value at the data point
    text_labels = []  # Create a list to store the text labels
    for i, cv in enumerate(data["CV"]):
        mean_value = data.iloc[i]["intervals_include_zero_mean"]
        text_labels.append(plt.text(cv, data.iloc[i]["intervals_include_zero_mean"], f"Mean: {mean_value:.4f}", fontsize=15, color=color))
    # Use adjust_text to adjust label positions and avoid overlap
    adjust_text(text_labels, force_explode = (0,0.5),  only_move = {'explode':'y'})
    


# handles, labels = plt.gca().get_legend_handles_labels()
# for method in methods:
#     se_value = df[df["MethodOfMoments"] == method]["intervals_include_zero_SE"].iloc[0]
#     plt.plot([], [], linestyle='', label=f"{method}, SE={se_value:.6f}", color='white')


plt.xlabel("CV")
plt.ylabel("ln ratio SE include zero")
plt.title("Grouped Line Chart by MethodOfMoments and N1")
plt.legend()
plt.grid(True)
plt.savefig("grouped_line_chart_Higgins1MoMNo_SE.png")
plt.show()

quit()

# Create a pivot table to group the data by (NoMethodOfMoments, N1)
pivot_table = df.pivot_table(index=["NoMethodOfMoments", "N1"], columns="CV", values="ln ratio SE include zero")

# Create a grouped line chart
plt.figure(figsize=(10, 6))
for (method, n1), data in pivot_table.groupby(level=[0, 1]):
    # Convert data.values to a NumPy array and flatten it
    values = np.array(data.values).flatten()
    plt.plot(data.index, values, marker='o', label=f"NoMethodOfMoments={method}, N1={n1}")

# Label the x-axis with CV values
x_labels = pivot_table.columns
plt.xticks(range(len(x_labels)), x_labels, rotation=45)

plt.xlabel("CV")
plt.ylabel("ln_ratio")
plt.title("Grouped Line Chart by NoMethodOfMoments and N1")
plt.legend()
plt.savefig("grouped_line_charts3.png")
plt.grid(True)
plt.show()