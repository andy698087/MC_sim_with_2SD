import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns  

# Create a DataFrame with your data, including N1=50
df = pd.read_excel('From_text_GPM_MC_nMonte_100000_N1_Higgins12MoMNo.xlsx')

col_new_names = []
for col in df.columns:
    col_new_names.append(col.strip())

df.columns = col_new_names
print(col_new_names)
df = df[df['N1'] != 2]
df = df[df['MethodOfMoments'] != 'Higgins2']
df = df[['MethodOfMoments','N1','CV', 'ln ratio SE include zero']]
print(df)
quit()

methods = [df["MethodOfMoments"].unique()[i] for i in range(len(df["MethodOfMoments"].unique()))]
# Define markers for each MethodOfMoments
# marker_styles = {methods[0]: 'o', methods[1]: 's', methods[2]: '^', methods[3]: 'D'}
marker_styles = {methods[0]: 'o', methods[1]: '^', methods[2]: 'D'}
colors = sns.color_palette("husl", n_colors=len(df["MethodOfMoments"].unique()) * len(df["N1"].unique()))



# Create a grouped line chart
plt.figure(figsize=(15, 9))
for (method, n1), data in df.groupby(["MethodOfMoments", "N1"]):
    color = colors.pop() # Get the next color from the palette
    marker_style = marker_styles.get(method, 'o')
    plt.plot(data["CV"], data["ln ratio SE include zero"], marker=marker_style, label=f"{method}, N1={n1}", color=color)

plt.xlabel("CV")
plt.ylabel("ln ratio SE include zero")
plt.title("Grouped Line Chart by MethodOfMoments and N1")
plt.legend()
plt.grid(True)
plt.savefig("grouped_line_chart_Higgins1MoMNo.png")
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