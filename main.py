import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def load_data(filepath):
    """Loads a CSV file and handles potential file errors."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found. Please check the file path.")
        return None

def clean_and_prepare_data(df1):
    """Cleans and preprocesses the bat landing dataset (dataset1)."""
    print("\n--- Cleaning and Preparing dataset1 ---")
    
    # Fill missing 'habit' values
    df1['habit'].fillna('unknown', inplace=True)
    
    # Convert date columns safely, turning errors into NaT (Not a Time)
    date_columns = ['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time']
    for col in date_columns:
        df1[col] = pd.to_datetime(df1[col], format='%d/%m/%Y %H:%M', errors='coerce')
    
    # Drop any rows where date conversion failed
    original_rows = len(df1)
    df1.dropna(subset=date_columns, inplace=True)
    print(f"Data cleaned. {original_rows - len(df1)} rows with invalid dates removed.")
    return df1

def engineer_features(df):
    """Creates new features to enhance analysis."""
    print("\n--- Engineering New Features ---")
    # Calculate the duration of rat presence in seconds
    df['rat_presence_duration'] = (df['rat_period_end'] - df['rat_period_start']).dt.total_seconds()
    print("New feature 'rat_presence_duration' created.")
    return df

def analyze_vigilance(df):
    """Performs EDA on bat vigilance based on risk."""
    print("\n--- EDA 1: Analyzing Bat Vigilance vs. Risk ---")
    
    # Provide a more comprehensive summary
    vigilance_summary = df.groupby('risk')['bat_landing_to_food'].agg(['mean', 'median', 'std'])
    print("\nStatistical Summary of Time to Approach Food (seconds):")
    print(vigilance_summary)
    
    # Visualize the distribution
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='risk', y='bat_landing_to_food', data=df)
    plt.title('Bat Vigilance: Time to Approach Food by Risk Behavior', fontsize=16)
    plt.xlabel('Risk Behavior (0 = Avoidance, 1 = Taking)', fontsize=12)
    plt.ylabel('Time from Landing to Food (seconds)', fontsize=12)
    plt.show()

def analyze_habit_by_risk(df):
    """Analyzes the frequency of habits for each risk group."""
    print("\n--- EDA 2: Analyzing Habit Frequencies by Risk Group ---")
    habit_summary = df.groupby('risk')['habit'].value_counts().unstack().fillna(0)
    print(habit_summary.T.sort_values(by=[0, 1], ascending=False))


def analyze_avoidance(df):
    """Performs EDA on bat avoidance using dataset2."""
    print("\n--- EDA 3: Analyzing Colony-Wide Avoidance ---")
    df['rat_presence'] = df['rat_minutes'].apply(lambda x: 'Rat Present' if x > 0 else 'No Rat')
    
    print("\nAverage Bat Landings per 30-min Interval:")
    print(df.groupby('rat_presence')['bat_landing_number'].mean())
    
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='rat_presence', y='bat_landing_number', data=df)
    plt.title('Colony-Wide Avoidance: Bat Landings vs. Rat Presence', fontsize=16)
    plt.xlabel('Rat Presence in Interval', fontsize=12)
    plt.ylabel('Number of Bat Landings', fontsize=12)
    plt.show()

def run_hypothesis_test(df):
    """Performs and interprets the t-test for statistical significance."""
    print("\n--- Hypothesis Test: Validating Vigilance Findings ---")
    risk_avoidance_group = df[df['risk'] == 0]['bat_landing_to_food']
    risk_taking_group = df[df['risk'] == 1]['bat_landing_to_food']
    
    t_stat, p_value = ttest_ind(risk_avoidance_group, risk_taking_group, nan_policy='omit')
    
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value}")
    
    if p_value < 0.05:
        print("Conclusion: The result is statistically significant (p < 0.05). We reject the null hypothesis.")
    else:
        print("Conclusion: The result is not statistically significant (p >= 0.05).")

def main():
    """Main function to orchestrate the data analysis workflow."""
    sns.set_theme(style="whitegrid", palette="viridis") # Sets a professional theme for all plots
    
    df1 = load_data('dataset1.csv')
    df2 = load_data('dataset2.csv')
    
    if df1 is not None and df2 is not None:
        df1 = clean_and_prepare_data(df1)
        df1 = engineer_features(df1)
        
        analyze_vigilance(df1)
        analyze_habit_by_risk(df1)
        analyze_avoidance(df2)
        run_hypothesis_test(df1)
        
        print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()