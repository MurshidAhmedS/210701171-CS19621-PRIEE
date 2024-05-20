import matplotlib.pyplot as plt

# Categories and corresponding counts
categories = ['Yes', 'No']
counts = [250, 400]  # Example counts for each category

# Plotting the bar graph
plt.bar(categories, counts, color=['blue', 'green'])

# Adding labels and title
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.title('Counts of Four Categories')

# Display the plot
plt.show()
