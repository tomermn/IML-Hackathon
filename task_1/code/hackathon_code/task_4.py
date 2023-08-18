import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming your data is stored in a DataFrame called 'df'


def plotCancellationDistribution(cancellation):
    top_10_codes = cancellation.value_counts().nlargest(10)

    plt.figure(figsize=(10, 8))
    bars = plt.bar(top_10_codes.index, top_10_codes.values)

    # Highlighting the top 1 code in gold
    bars[0].set_color('gold')

    plt.xticks(rotation=90)
    plt.xlabel('Cancellation Policy Code')
    plt.ylabel('Count')
    plt.title('Top 10 Cancellation Policy Codes')
    plt.tight_layout()
    plt.show()


def plotHotelRatingCancellation(X: pd.DataFrame, y: pd.DataFrame):
    rating_counts = {'0': 0, '0.5': 0, '1': 0, '1,5': 0, '2': 0, '2.5': 0, '3': 0, '3.5': 0, '4': 0, '4.5': 0, '5': 0}
    valid_index = X.index
    for i in valid_index:
        if y[i] == 1:
            rating_counts[str(int(X['hotel_star_rating'][i]))] += 1

    # Sort the dictionary by the hotel_star_rating in ascending order
    sorted_ratings = sorted(rating_counts.items())

    # Extract the ratings and counts into separate lists
    ratings = [rating for rating, count in sorted_ratings]
    counts = [count for rating, count in sorted_ratings]

    # Plot the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(ratings, counts)

    plt.xlabel('Hotel Star Rating')
    plt.ylabel('Number of Cancellations')
    plt.title('Cancellations by Hotel Star Rating')

    plt.xticks(ratings)
    plt.show()


def get_intervals(X: pd.DataFrame):
    advanced_order_days = X["advanced_order_days"]

    # Find the maximum and minimum values
    max_value = advanced_order_days.max()
    min_value = advanced_order_days.min()

    # Calculate the interval length
    interval_length = (max_value - min_value) / 8

    # Generate the equally spaced values within the interval
    sliced_values = np.arange(min_value, max_value + interval_length, interval_length)
    lst = []
    for i in range(len(sliced_values) - 1):
        lst.append([sliced_values[i], sliced_values[i+1], 0])
    return lst


def cancellationPolicy(X: pd.DataFrame, y, cancellation):
    plotCancellationDistribution(cancellation)
    plotHotelRatingCancellation(X, y)
    intervals = get_intervals(X)
    valid_index = X.index
    total_cancellation = 0
    for i in valid_index:
        if y[i] == 1:
            for interval in intervals:
                if interval[0] <= X['advanced_order_days'][i] <= interval[1]:
                    interval[2] += 1
                    total_cancellation += 1
    percentages = [(interval[2] / total_cancellation) * 100 for interval in intervals]

    # Generate labels for the pie chart with exact interval values
    labels = [f"Orders between {int(interval[0])} - {int(interval[1])} days before checkin" for interval in intervals]

    # Plot the pie chart with a larger figure size and spacing between slices and labels
    plt.figure(figsize=(14, 14))
    plt.pie(percentages, labels=labels, autopct="%1.1f%%", labeldistance=1)
    plt.title("Cancellation Percentage by Interval")
    plt.axis("equal")  # Ensure a circular pie chart

    # Increase the spacing of the label 'order between 407 - 444 days before checkin'
    plt.show()


def calculate_cancellation_cost(days_before_checkin, factor, total_cost):
    if days_before_checkin > 20:
        return 0
    return (4 / (days_before_checkin * factor))*total_cost