import pandas as pd
import dash
from dash import dcc, html, Input, Output, dash_table, callback
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import numpy as np
import os
import re
from dash.exceptions import PreventUpdate

# Configuration
CSV_FILE_PATH = 'bank_statements_processed.csv'
CONFIG_FILE_PATH = 'settings.conf'


def load_config(filepath):
    """Load configuration including categories, allocations, budget percentages, and known merchants"""
    default_config = {
        "categories": {
            "Groceries": ["grocery", "supermarket", "food", "market", "walmart", "target", "costco"],
            "Dining": ["restaurant", "cafe", "coffee", "pizza", "burger", "dining", "doordash", "ubereats"],
            "Transportation": ["gas", "fuel", "uber", "lyft", "parking", "transit", "metro"],
            "Utilities": ["electric", "water", "gas company", "internet", "phone", "utility"],
            "Entertainment": ["netflix", "spotify", "movie", "theater", "game", "entertainment"],
            "Shopping": ["amazon", "store", "shop", "retail", "clothing"],
            "Healthcare": ["pharmacy", "doctor", "hospital", "medical", "health"],
            "Subscriptions": ["tmobile", "education", "subscription", "monthly"],
            "Transfer": ["transfer", "deposit", "withdrawal", "atm", "internal"]
        },
        "allocations": {
            "Groceries": "needs",
            "Dining": "wants",
            "Transportation": "needs",
            "Utilities": "needs",
            "Entertainment": "wants",
            "Shopping": "wants",
            "Healthcare": "needs",
            "Subscriptions": "wants",
            "Transfer": "saving"
        },
        "budget_percentages": {
            "needs": 50,
            "wants": 20,
            "saving": 30
        },
        "known_merchants": [
            "netflix", "spotify", "amazon", "apple", "google", "microsoft",
            "walmart", "target", "costco", "starbucks", "uber", "lyft",
            "paypal", "venmo", "square", "chase", "wells fargo", "bank of america",
            "disney", "hulu", "youtube", "adobe", "dropbox", "slack", "zoom",
            "netflix", "hbo", "discovery", "paramount", "peacock", "hulu",
            "apple music", "tidal", "pandora", "soundcloud", "amazon prime",
            "equifax", "experian", "transunion", "credit karma", "mint",
            "verizon", "att", "t-mobile", "sprint", "xfinity", "comcast",
            "shell", "chevron", "exxon", "mobil", "bp", "marathon",
            "mcdonalds", "subway", "kfc", "wendys", "taco bell", "chipotle",
            "best buy", "home depot", "lowes", "ikea", "sears", "kohls",
            "nordstrom", "macys", "jcpenney", "kohl's", "ulta", "sephora"
        ]
    }

    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
                # Merge with defaults to ensure all keys exist
                merged_config = default_config.copy()
                merged_config.update(config)
                merged_config["categories"].update(config.get("categories", {}))
                merged_config["allocations"].update(config.get("allocations", {}))
                merged_config["budget_percentages"].update(config.get("budget_percentages", {}))
                merged_config["known_merchants"].extend(config.get("known_merchants", []))
                # Remove duplicates while preserving order
                merged_config["known_merchants"] = list(dict.fromkeys(merged_config["known_merchants"]))
                return merged_config
        except Exception as e:
            print(f"Error loading configuration file: {e}")

    print("Using default configuration")
    return default_config


def categorize_transaction(description):
    """Simple categorization based on description keywords"""
    if pd.isna(description):
        return 'Other'

    description_lower = str(description).lower()

    for category, keywords in CONFIG["categories"].items():
        if any(keyword.lower() in description_lower for keyword in keywords):
            return category

    return 'Other'


def get_allocation(category):
    """Get allocation type for a category"""
    return CONFIG["allocations"].get(category, "other")


def normalize_description(desc, known_merchants):
    """Normalize transaction descriptions to identify same merchants with different names"""
    if pd.isna(desc):
        return 'Unknown'

    desc = str(desc).strip().lower()

    # Common patterns to extract merchant name
    # Remove common words that don't identify the merchant
    desc = re.sub(r'\s+', ' ', desc)  # Replace multiple spaces with single space
    desc = re.sub(r'purchase\s*on\s*\d{2}/\d{2}/\d{4}', '', desc)  # Remove purchase date
    desc = re.sub(r'purchase\s*at', '', desc)  # Remove purchase at
    desc = re.sub(r'payment.*', '', desc)  # Remove payment references
    desc = re.sub(r'\d+\.\d{2}$', '', desc)  # Remove trailing amounts
    desc = re.sub(r'[-_.,]', ' ', desc)  # Replace separators with spaces

    # Look for known merchant patterns first
    for merchant in known_merchants:
        if merchant in desc:
            return merchant

    # If no known merchant found, try to extract merchant from the description
    # Remove common prefixes/suffixes
    desc = re.sub(r'^\*|^*\s|\s\*$', '', desc)  # Remove leading/trailing asterisks
    desc = re.sub(r'^www\.|\.com$|\.org$|\.net$|\.io$', '', desc)  # Remove domain parts
    desc = re.sub(r'\s+', ' ', desc).strip()  # Clean up whitespace again

    # Split by common separators and take the first meaningful part
    parts = re.split(r'[,\\-\\|]', desc)
    for part in parts:
        part = part.strip()
        if len(part) > 2 and part not in ['online', 'web', 'pos', 'debit', 'credit', 'purchase', 'payment']:
            return part

    # If still no clear merchant, return the cleaned description
    return 'Other'


def load_data(filepath):
    """Load and parse CSV file from disk with better error handling"""
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist!")
        return None

    try:
        df = pd.read_csv(filepath)

        # Validate required columns exist
        required_columns = ['Date', 'Description']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"Missing required columns: {missing}")
            return None

        # Handle different column naming conventions
        debit_col = 'Debit' if 'Debit' in df.columns else 'Amount' if 'Amount' in df.columns else None
        credit_col = 'Credit' if 'Credit' in df.columns else None

        if debit_col:
            df['Debit'] = pd.to_numeric(df[debit_col], errors='coerce').fillna(0)
        else:
            df['Debit'] = 0

        if credit_col:
            df['Credit'] = pd.to_numeric(df[credit_col], errors='coerce').fillna(0)
        else:
            df['Credit'] = 0

        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Remove rows with invalid dates
        df = df.dropna(subset=['Date'])

        # Add Category column
        df['Category'] = df['Description'].apply(categorize_transaction)

        # Add Allocation column
        df['Allocation'] = df['Category'].apply(get_allocation)

        # Normalize descriptions to group similar merchants
        df['Normalized_Description'] = df['Description'].apply(
            lambda x: normalize_description(x, CONFIG["known_merchants"])
        )

        # Add transaction type based on amount
        df['Transaction Type'] = df.apply(
            lambda row: 'Credit' if row['Credit'] > 0 else 'Debit' if row['Debit'] > 0 else 'Other',
            axis=1
        )

        return df

    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


# Load configuration and data
CONFIG = load_config(CONFIG_FILE_PATH)
df_global = load_data(CSV_FILE_PATH)


def filter_spending_data(df):
    """Filter out credit and transfer transactions for spending analysis"""
    # Only keep debit transactions that are not transfers
    return df[(df['Transaction Type'] == 'Debit') & (df['Category'] != 'Transfer')]


def identify_recurring_transactions(df_spending, threshold=50, min_frequency=3):
    """
    Identify potential recurring transactions (death by a thousand cuts)
    Groups similar merchants with different descriptions
    """
    # Group by normalized description and count occurrences
    transaction_counts = df_spending.groupby('Normalized_Description').size().reset_index(name='Frequency')

    # Calculate average amount per normalized description
    transaction_avg = df_spending.groupby('Normalized_Description')['Debit'].mean().reset_index()
    transaction_avg.rename(columns={'Debit': 'Avg_Amount'}, inplace=True)

    # Calculate total amount per normalized description
    transaction_total = df_spending.groupby('Normalized_Description')['Debit'].sum().reset_index()
    transaction_total.rename(columns={'Debit': 'Total_Amount'}, inplace=True)

    # Get sample descriptions for each normalized group
    sample_descriptions = df_spending.groupby('Normalized_Description')['Description'].first().reset_index()
    sample_descriptions.rename(columns={'Description': 'Sample_Description'}, inplace=True)

    # Combine counts, averages, totals, and sample descriptions
    recurring_df = transaction_counts.merge(transaction_avg, on='Normalized_Description')
    recurring_df = recurring_df.merge(transaction_total, on='Normalized_Description')
    recurring_df = recurring_df.merge(sample_descriptions, on='Normalized_Description')

    # Filter for transactions that occur frequently and have low average amounts
    death_by_thousand_cuts = recurring_df[
        (recurring_df['Frequency'] >= min_frequency) &
        (recurring_df['Avg_Amount'] <= threshold)
        ].sort_values(by=['Total_Amount'], ascending=False)

    return death_by_thousand_cuts


def calculate_budget_summary(df_filtered):
    """Calculate budget summary with actual vs percentage allocations"""
    if df_filtered.empty:
        return [], {}

    # Filter to only spending transactions
    df_spending = filter_spending_data(df_filtered)
    total_spending = df_spending['Debit'].sum()

    if total_spending == 0:
        return [], {"total_spending": 0}

    # Group by allocation type and sum spending
    allocation_spending = df_spending.groupby('Allocation')['Debit'].sum().to_dict()

    # Get budget percentages
    budget_percentages = CONFIG["budget_percentages"]

    # Calculate actual percentages
    allocation_percentages = {}
    for alloc_type in budget_percentages.keys():
        spent = allocation_spending.get(alloc_type, 0)
        actual_percentage = (spent / total_spending * 100) if total_spending > 0 else 0
        allocation_percentages[alloc_type] = actual_percentage

    return allocation_spending, {
        "total_spending": total_spending,
        "allocation_spending": allocation_spending,
        "budget_percentages": budget_percentages,
        "actual_percentages": allocation_percentages
    }


def create_budget_summary(df_filtered):
    """Create budget summary with actual vs percentage allocations"""
    if df_filtered.empty:
        return [html.Div("No data available for selected period", className="metric-card")]

    allocation_spending, summary_data = calculate_budget_summary(df_filtered)

    if summary_data["total_spending"] == 0:
        return [html.Div("No spending data for selected period", className="metric-card")]

    # Create budget cards
    budget_cards = []
    for alloc_type in ['needs', 'wants', 'saving']:
        spent = summary_data["allocation_spending"].get(alloc_type, 0)
        budget_percentage = summary_data["budget_percentages"].get(alloc_type, 0)
        actual_percentage = summary_data["actual_percentages"].get(alloc_type, 0)

        # Calculate budget amount based on total spending
        budget_amount = (budget_percentage / 100) * summary_data["total_spending"]
        variance = actual_percentage - budget_percentage

        # Determine color based on variance
        if abs(variance) <= 2:  # Within 2% tolerance
            card_color = "#27ae60"  # Green
            bg_color = "#d5f4e6"
        elif variance > 0:  # Over budget
            card_color = "#e74c3c"  # Red
            bg_color = "#ffd6d6"
        else:  # Under budget
            card_color = "#3498db"  # Blue
            bg_color = "#d6eaf8"

        budget_cards.append(
            html.Div([
                html.H4(f"{alloc_type.title()}", style={'color': '#7f8c8d', 'fontSize': '16px'}),
                html.H3(f"${spent:,.2f}", style={'color': card_color, 'margin': '5px 0'}),
                html.P(f"Amt: ${budget_amount:,.2f} ({budget_percentage:.0f}%)",
                       style={'margin': '2px 0', 'fontSize': '12px'}),
                html.P(f"Actual: {actual_percentage:.1f}%",
                       style={'margin': '2px 0', 'fontSize': '12px'}),
                html.P(f"Variance: {variance:+.1f}%",
                       style={'margin': '2px 0', 'fontSize': '12px'}),
                html.Div([
                    html.Span(f"{actual_percentage:.1f}% / {budget_percentage:.1f}%",
                              style={'fontSize': '12px', 'fontWeight': 'bold'}),
                    html.Div(style={
                        'height': '6px',
                        'backgroundColor': '#ddd',
                        'borderRadius': '3px',
                        'marginTop': '5px'
                    }, children=[
                        html.Div(style={
                            'height': '100%',
                            'width': f'{min(max(actual_percentage, 0), 100)}%',
                            'backgroundColor': card_color if actual_percentage <= budget_percentage else '#e74c3c',
                            'borderRadius': '3px'
                        })
                    ])
                ], style={'marginTop': '5px'})
            ], className="budget-card",
                style={'backgroundColor': bg_color, 'border': f'1px solid {card_color}'})
        )

    return budget_cards


def create_death_by_thousand_cuts_chart(df_filtered):
    """Create a visualization of recurring small transactions"""
    if df_filtered.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig

    # Filter to only spending transactions
    df_spending = filter_spending_data(df_filtered)

    if df_spending.empty:
        fig = go.Figure()
        fig.add_annotation(text="No spending data", showarrow=False)
        return fig

    # Identify recurring transactions
    recurring_transactions = identify_recurring_transactions(df_spending)

    if recurring_transactions.empty:
        fig = go.Figure()
        fig.add_annotation(text="No recurring small transactions found", showarrow=False)
        return fig

    # Take top 10 recurring transactions by total amount
    top_recurring = recurring_transactions.head(10)

    # Create ranks for the x-axis
    ranks = list(range(1, len(top_recurring) + 1))

    # Create a bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=ranks,
            y=top_recurring['Total_Amount'],
            name='Total Amount',
            marker_color='lightcoral',
            text=top_recurring['Normalized_Description'],
            textposition='outside',
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                          'Merchant: %{text}<br>' +
                          'Total Amount: $%{y:.2f}<br>' +
                          'Average: $%{customdata[1]:.2f}<br>' +
                          'Frequency: %{customdata[2]}<br>' +
                          '<extra></extra>',
            customdata=list(
                zip(top_recurring['Sample_Description'], top_recurring['Avg_Amount'], top_recurring['Frequency']))
        )
    ])

    fig.update_layout(
        title=f'Top Recurring Small Transactions by Total Amount<br>Total Potential Savings: ${top_recurring["Total_Amount"].sum():.2f}',
        xaxis_title='Rank',
        yaxis_title='Total Amount ($)',
        xaxis=dict(
            tickmode='array',
            tickvals=ranks,
            ticktext=[f'#{i}' for i in ranks]
        ),
        template='plotly_white',
        height=500,
        showlegend=False
    )

    return fig


def create_allocation_breakdown(df_filtered):
    """Create allocation breakdown visualization"""
    if df_filtered.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig

    # Calculate budget summary
    allocation_spending, summary_data = calculate_budget_summary(df_filtered)

    if not summary_data["allocation_spending"]:
        fig = go.Figure()
        fig.add_annotation(text="No spending data", showarrow=False)
        return fig

    # Create comparison chart showing both budgeted and actual percentages
    allocation_types = list(summary_data["budget_percentages"].keys())
    budget_percentages = [summary_data["budget_percentages"].get(alloc, 0) for alloc in allocation_types]
    actual_percentages = [summary_data["actual_percentages"].get(alloc, 0) for alloc in allocation_types]

    # Create grouped bar chart
    fig = go.Figure(data=[
        go.Bar(name='Budgeted %', x=allocation_types, y=budget_percentages,
               marker_color='#3498db', opacity=0.7),
        go.Bar(name='Actual %', x=allocation_types, y=actual_percentages,
               marker_color='#e74c3c', opacity=0.7)
    ])

    fig.update_layout(
        title='Budgeted vs Actual Allocation Percentages',
        xaxis_title='Allocation Type',
        yaxis_title='Percentage (%)',
        barmode='group',
        template='plotly_white',
        height=400
    )

    return fig


def create_metrics_cards(df_filtered):
    """Create metric cards based on filtered data"""
    if df_filtered.empty:
        return [html.Div("No data available for selected period", className="metric-card")]

    # Filter to only spending transactions
    df_spending = filter_spending_data(df_filtered)

    # Calculate metrics
    total_debits = df_spending['Debit'].sum()
    total_credits = df_filtered[df_filtered['Transaction Type'] == 'Credit']['Credit'].sum()
    net_cash_flow = total_credits - df_filtered['Debit'].sum()  # Include all debits for cash flow
    avg_transaction = df_spending[df_spending['Debit'] > 0]['Debit'].mean() if any(df_spending['Debit'] > 0) else 0
    transaction_count = len(df_spending)  # Only count spending transactions

    # Identify death by thousand cuts
    recurring_transactions = identify_recurring_transactions(df_spending)
    potential_savings = recurring_transactions['Total_Amount'].sum() if not recurring_transactions.empty else 0

    metrics = [
        html.Div([
            html.H4("Total Spending", style={'color': '#7f8c8d', 'fontSize': '16px'}),
            html.H2(f"${total_debits:,.2f}", style={'color': '#e74c3c', 'margin': '10px 0'})
        ], className="metric-card"),

        html.Div([
            html.H4("Total Income", style={'color': '#7f8c8d', 'fontSize': '16px'}),
            html.H2(f"${total_credits:,.2f}", style={'color': '#27ae60', 'margin': '10px 0'})
        ], className="metric-card"),

        html.Div([
            html.H4("Net Cash Flow", style={'color': '#7f8c8d', 'fontSize': '16px'}),
            html.H2(f"${net_cash_flow:,.2f}",
                    style={'color': '#27ae60' if net_cash_flow > 0 else '#e74c3c', 'margin': '10px 0'})
        ], className="metric-card"),

        html.Div([
            html.H4("Avg Spending", style={'color': '#7f8c8d', 'fontSize': '16px'}),
            html.H2(f"${avg_transaction:,.2f}", style={'color': '#3498db', 'margin': '10px 0'})
        ], className="metric-card"),

        html.Div([
            html.H4("Potential Savings", style={'color': '#7f8c8d', 'fontSize': '16px'}),
            html.H2(f"${potential_savings:,.2f}", style={'color': '#9b59b6', 'margin': '10px 0'})
        ], className="metric-card"),
    ]

    return metrics


def create_spending_over_time(df_filtered):
    """Create spending over time visualization"""
    if df_filtered.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig

    # Filter to only spending transactions
    df_spending = filter_spending_data(df_filtered)

    # Daily aggregation
    daily_spending = df_spending.groupby(df_spending['Date'].dt.date).agg({'Debit': 'sum'}).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_spending['Date'],
        y=daily_spending['Debit'],
        mode='lines+markers',
        name='Spending',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title='Daily Spending',
        xaxis_title='Date',
        yaxis_title='Amount ($)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def create_category_breakdown(df_filtered):
    """Create category breakdown visualization"""
    if df_filtered.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig

    # Filter to only spending transactions
    df_spending = filter_spending_data(df_filtered)

    if df_spending.empty:
        fig = go.Figure()
        fig.add_annotation(text="No spending data", showarrow=False)
        return fig

    category_spending = df_spending[df_spending['Debit'] > 0].groupby('Category')['Debit'].sum().sort_values(
        ascending=True)

    if category_spending.empty:
        fig = go.Figure()
        fig.add_annotation(text="No debit transactions", showarrow=False)
        return fig

    fig = px.bar(
        x=category_spending.values,
        y=category_spending.index,
        orientation='h',
        title='Spending by Category',
        labels={'x': 'Amount ($)', 'y': 'Category'},
        color=category_spending.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(
        showlegend=False,
        template='plotly_white',
        height=400
    )

    return fig


def create_category_spending_distribution(df_filtered):
    """Create category spending distribution pie chart"""
    if df_filtered.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig

    # Filter to only spending transactions
    df_spending = filter_spending_data(df_filtered)

    if df_spending.empty:
        fig = go.Figure()
        fig.add_annotation(text="No spending data", showarrow=False)
        return fig

    # Group by category and sum spending
    category_spending = df_spending[df_spending['Debit'] > 0].groupby('Category')['Debit'].sum()

    if category_spending.empty:
        fig = go.Figure()
        fig.add_annotation(text="No debit transactions", showarrow=False)
        return fig

    # Create pie chart
    fig = px.pie(
        values=category_spending.values,
        names=category_spending.index,
        title='Spending Distribution by Category',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label+value')
    fig.update_layout(template='plotly_white', height=400)

    return fig


# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Main layout
app.layout = html.Div([
    html.Div([
        html.H1("Financial Analysis Dashboard",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30, 'fontFamily': 'Arial'}),

        # Status message
        html.Div(id='status-message', style={'textAlign': 'center', 'color': '#e74c3c', 'marginBottom': 20}),

        # Controls container
        html.Div([
            # Date range selector
            html.Div([
                html.Label("Select Date Range:",
                           style={'fontWeight': 'bold', 'marginRight': 10, 'verticalAlign': 'middle'}),
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date=df_global['Date'].min() if df_global is not None else datetime.now() - timedelta(
                        days=30),
                    end_date=df_global['Date'].max() if df_global is not None else datetime.now(),
                    display_format='YYYY-MM-DD',
                    style={'verticalAlign': 'middle'}
                )
            ], style={'display': 'inline-block', 'marginRight': 30}),

            # Category filter
            html.Div([
                html.Label("Filter by Category:",
                           style={'fontWeight': 'bold', 'marginRight': 10, 'verticalAlign': 'middle'}),
                dcc.Dropdown(
                    id='global-category-filter',
                    options=[],
                    value='All',
                    style={'width': '200px', 'verticalAlign': 'middle'}
                )
            ], style={'display': 'inline-block'}),

            # Allocation filter
            html.Div([
                html.Label("Filter by Allocation:",
                           style={'fontWeight': 'bold', 'marginRight': 10, 'verticalAlign': 'middle'}),
                dcc.Dropdown(
                    id='allocation-filter',
                    options=[{'label': 'All', 'value': 'All'},
                             {'label': 'Needs', 'value': 'needs'},
                             {'label': 'Wants', 'value': 'wants'},
                             {'label': 'Saving', 'value': 'saving'},
                             {'label': 'Other', 'value': 'other'}],
                    value='All',
                    style={'width': '200px', 'verticalAlign': 'middle'}
                )
            ], style={'display': 'inline-block'})

        ], style={'textAlign': 'center', 'marginBottom': 20}),

        # Budget summary cards
        html.Div(id='budget-summary',
                 style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap', 'gap': '10px',
                        'marginBottom': 30}),

        # Metrics cards
        html.Div(id='metrics-cards',
                 style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap', 'gap': '10px',
                        'marginBottom': 30}),

        # Charts
        html.Div([
            # Row 1
            html.Div([
                html.Div([dcc.Graph(id='spending-over-time')], style={'width': '100%'}),
            ], style={'width': '100%', 'marginBottom': 20}),

            # Row 2
            html.Div([
                html.Div([dcc.Graph(id='category-breakdown')], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='allocation-breakdown')],
                         style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
            ], style={'width': '100%', 'marginBottom': 20}),

            # Row 3
            html.Div([
                html.Div([dcc.Graph(id='category-spending-distribution')],
                         style={'width': '48%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='death-by-thousand-cuts')],
                         style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
            ], style={'width': '100%', 'marginBottom': 20}),

            # Row 4
            html.Div([
                html.Div([dcc.Graph(id='debit-distribution')], style={'width': '100%'}),
            ], style={'width': '100%', 'marginBottom': 30}),
        ]),

        # Top debits table
        html.Div([
            html.H3("Top Spending", style={'textAlign': 'center', 'marginTop': 30, 'marginBottom': 20}),

            html.Div([
                html.Div([
                    html.Label("Category Filter:", style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.Dropdown(
                        id='table-category-filter',
                        options=[],
                        value='All',
                        style={'width': '200px'}
                    )
                ], style={'display': 'inline-block', 'marginRight': 30}),

                html.Div([
                    html.Label("Allocation Filter:", style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.Dropdown(
                        id='table-allocation-filter',
                        options=[{'label': 'All', 'value': 'All'},
                                 {'label': 'Needs', 'value': 'needs'},
                                 {'label': 'Wants', 'value': 'wants'},
                                 {'label': 'Saving', 'value': 'saving'},
                                 {'label': 'Other', 'value': 'other'}],
                        value='All',
                        style={'width': '200px'}
                    )
                ], style={'display': 'inline-block', 'marginRight': 30}),

                html.Div([
                    html.Label("Search Description:", style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.Input(
                        id='description-filter',
                        type='text',
                        placeholder='Enter keywords...',
                        style={'width': '300px', 'padding': '5px'}
                    )
                ], style={'display': 'inline-block'})
            ], style={'textAlign': 'center', 'marginBottom': 20}),

            html.Div(id='top-transactions-table', style={'margin': '0 20px'})
        ], style={'marginTop': 30, 'marginBottom': 30})

    ], style={'padding': '20px', 'maxWidth': '1600px', 'margin': '0 auto'})
], style={'fontFamily': 'Arial'})

# CSS for metric cards and budget cards
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .metric-card {
                textAlign: center;
                padding: 20px;
                backgroundColor: #ecf0f1;
                borderRadius: 8px;
                width: 18%;
                minWidth: 180px;
                boxShadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .budget-card {
                textAlign: center;
                padding: 15px;
                borderRadius: 8px;
                width: 23%;
                minWidth: 180px;
                boxShadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            @media (max-width: 768px) {
                .metric-card, .budget-card {
                    width: 45%;
                    margin-bottom: 10px;
                }
            }
            @media (max-width: 480px) {
                .metric-card, .budget-card {
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


# Callbacks
@app.callback(
    Output('status-message', 'children'),
    Input('date-picker', 'start_date')
)
def update_status_message(start_date):
    if df_global is None:
        return "❌ Error: No data loaded. Please check your CSV file path and format."

    # Show only spending transactions info
    df_spending = filter_spending_data(df_global)
    return f"✅ Loaded {len(df_global):,} transactions ({len(df_spending):,} spending) from {df_global['Date'].min().strftime('%Y-%m-%d')} to {df_global['Date'].max().strftime('%Y-%m-%d')}"


@app.callback(
    [Output('global-category-filter', 'options'),
     Output('table-category-filter', 'options')],
    Input('date-picker', 'start_date')
)
def update_category_dropdowns(start_date):
    """Populate category dropdowns with available categories"""
    if df_global is None:
        return [], []

    # Only show categories from spending transactions
    df_spending = filter_spending_data(df_global)
    categories = ['All'] + sorted(df_spending['Category'].unique().tolist())
    options = [{'label': cat, 'value': cat} for cat in categories]

    return options, options


@app.callback(
    [Output('budget-summary', 'children'),
     Output('metrics-cards', 'children'),
     Output('spending-over-time', 'figure'),
     Output('category-breakdown', 'figure'),
     Output('allocation-breakdown', 'figure'),
     Output('category-spending-distribution', 'figure'),
     Output('death-by-thousand-cuts', 'figure'),
     Output('debit-distribution', 'figure')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('global-category-filter', 'value'),
     Input('allocation-filter', 'value')]
)
def update_dashboard_charts(start_date, end_date, selected_category, selected_allocation):
    if df_global is None:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Error loading data", showarrow=False)
        return [], [], empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    # Filter by date range
    df = df_global.copy()
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_filtered = df[mask].copy()

    # Apply category filter if not 'All'
    if selected_category and selected_category != 'All':
        df_filtered = df_filtered[df_filtered['Category'] == selected_category]

    # Apply allocation filter if not 'All'
    if selected_allocation and selected_allocation != 'All':
        df_filtered = df_filtered[df_filtered['Allocation'] == selected_allocation]

    # Create visualizations
    budget_cards = create_budget_summary(df_filtered)
    metrics = create_metrics_cards(df_filtered)
    spending_fig = create_spending_over_time(df_filtered)
    category_bar_fig = create_category_breakdown(df_filtered)
    allocation_comparison_fig = create_allocation_breakdown(df_filtered)
    category_spending_pie_fig = create_category_spending_distribution(df_filtered)
    death_by_thousand_cuts_fig = create_death_by_thousand_cuts_chart(df_filtered)

    # Debit distribution (only spending transactions)
    df_spending = filter_spending_data(df_filtered)
    if not df_spending.empty:
        df_debits = df_spending[df_spending['Debit'] > 0].copy().sort_values('Debit', ascending=False).reset_index(
            drop=True)
        df_debits['Rank'] = range(1, len(df_debits) + 1)

        debit_dist_fig = go.Figure()
        debit_dist_fig.add_trace(go.Scatter(
            x=df_debits['Rank'],
            y=df_debits['Debit'],
            mode='markers',
            name='Spending',
            marker=dict(size=8, color='red', opacity=0.6),
            text=df_debits['Description'],
            hovertemplate='<b>%{text}</b><br>Amount: $%{y:.2f}<br>Rank: %{x}<extra></extra>'
        ))
        debit_dist_fig.update_layout(
            title='Spending Distribution (Highest to Lowest)',
            xaxis_title='Rank',
            yaxis_title='Amount ($)',
            template='plotly_white',
            height=400
        )
    else:
        debit_dist_fig = go.Figure()
        debit_dist_fig.add_annotation(text="No data available", showarrow=False)

    return budget_cards, metrics, spending_fig, category_bar_fig, allocation_comparison_fig, category_spending_pie_fig, death_by_thousand_cuts_fig, debit_dist_fig


@app.callback(
    Output('top-transactions-table', 'children'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('table-category-filter', 'value'),
     Input('table-allocation-filter', 'value'),
     Input('description-filter', 'value')]
)
def update_transactions_table(start_date, end_date, selected_category, selected_allocation, description_search):
    """Update the top transactions table based on filters"""
    if df_global is None:
        return html.Div("No data available")

    df = df_global.copy()

    # Filter to only spending transactions
    df = filter_spending_data(df)

    # Apply date filter
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_filtered = df[mask].copy()

    # Apply category filter
    if selected_category and selected_category != 'All':
        df_filtered = df_filtered[df_filtered['Category'] == selected_category]

    # Apply allocation filter
    if selected_allocation and selected_allocation != 'All':
        df_filtered = df_filtered[df_filtered['Allocation'] == selected_allocation]

    # Apply description search
    if description_search and description_search.strip():
        df_filtered = df_filtered[
            df_filtered['Description'].str.contains(description_search, case=False, na=False)
        ]

    # Select relevant columns and sort by amount
    top_transactions = df_filtered.nlargest(10, 'Debit')[
        ['Date', 'Description', 'Category', 'Allocation', 'Debit', 'Transaction Type']
    ].copy()
    top_transactions['Amount'] = top_transactions['Debit']

    # Add merchant column
    top_transactions['Merchant'] = top_transactions['Description'].apply(
        lambda x: normalize_description(x, CONFIG["known_merchants"])
    )

    # Format date and amount
    top_transactions['Date'] = top_transactions['Date'].dt.strftime('%Y-%m-%d')
    top_transactions['Amount'] = top_transactions['Amount'].apply(lambda x: f'${x:,.2f}')

    # Create table
    table = dash_table.DataTable(
        data=top_transactions.to_dict('records'),
        columns=[
            {'name': 'Date', 'id': 'Date'},
            {'name': 'Description', 'id': 'Description'},
            {'name': 'Merchant', 'id': 'Merchant'},
            {'name': 'Category', 'id': 'Category'},
            {'name': 'Allocation', 'id': 'Allocation'},
            {'name': 'Amount', 'id': 'Amount'},
            {'name': 'Type', 'id': 'Transaction Type'}
        ],
        page_size=10,
        sort_action="native",
        filter_action="native",
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontSize': '14px'
        },
        style_header={
            'backgroundColor': '#3498db',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'left'
        },
        style_data={
            'backgroundColor': '#f8f9fa',
            'border': '1px solid #dee2e6'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#ffffff'
            },
            {
                'if': {
                    'filter_query': '{Allocation} = needs',
                    'column_id': 'Allocation'
                },
                'backgroundColor': '#d5f4e6'
            },
            {
                'if': {
                    'filter_query': '{Allocation} = wants',
                    'column_id': 'Allocation'
                },
                'backgroundColor': '#ffd6d6'
            },
            {
                'if': {
                    'filter_query': '{Allocation} = saving',
                    'column_id': 'Allocation'
                },
                'backgroundColor': '#d6eaf8'
            },
            {
                'if': {
                    'column_id': 'Amount'
                },
                'color': '#e74c3c'  # All amounts are spending, so always red
            },
            {
                'if': {
                    'filter_query': '{Merchant} = Other',
                    'column_id': 'Merchant'
                },
                'backgroundColor': '#f8f9fa',
                'color': '#95a5a6'
            }
        ]
    )

    return table


if __name__ == '__main__':
    app.run(debug=True, port=8050)