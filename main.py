import pandas as pd
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import numpy as np

# CSV file path - UPDATE THIS PATH TO YOUR CSV FILE
CSV_FILE_PATH = 'bank_statements.csv'

# Configuration file path
CONFIG_FILE_PATH = 'settings.conf'


# Load categories from configuration file
def load_categories(filepath):
    """Load category keywords from configuration file"""
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
            return config.get('categories', {})
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        print("Using default categories")
        # Default categories if file cannot be loaded
        return {
            'Groceries': ['grocery', 'supermarket', 'food', 'market', 'walmart', 'target', 'costco'],
            'Dining': ['restaurant', 'cafe', 'coffee', 'pizza', 'burger', 'dining', 'doordash', 'ubereats'],
            'Transportation': ['gas', 'fuel', 'uber', 'lyft', 'parking', 'transit', 'metro'],
            'Utilities': ['electric', 'water', 'gas company', 'internet', 'phone', 'utility'],
            'Entertainment': ['netflix', 'spotify', 'movie', 'theater', 'game', 'entertainment'],
            'Shopping': ['amazon', 'store', 'shop', 'retail', 'clothing'],
            'Healthcare': ['pharmacy', 'doctor', 'hospital', 'medical', 'health'],
            'Subscriptions': ['tmobile', 'education'],
            'Transfer': ['transfer', 'deposit', 'withdrawal', 'atm'],
        }


# Load categories at startup
CATEGORIES = load_categories(CONFIG_FILE_PATH)


def categorize_transaction(description):
    """Simple categorization based on description keywords"""
    description = str(description).lower()

    for category, keywords in CATEGORIES.items():
        if any(keyword in description for keyword in keywords):
            return category

    return 'Other'


# Load data from CSV file
def load_data(filepath):
    """Load and parse CSV file from disk"""
    try:
        df = pd.read_csv(filepath)
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        # Convert Debit and Credit to numeric, replacing any non-numeric values with 0
        df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
        df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)
        # Convert Account Running Balance to numeric as well
        df['Account Running Balance'] = pd.to_numeric(df['Account Running Balance'], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


# Load the data at startup
df_global = load_data(CSV_FILE_PATH)

# Add categories to the dataframe and save as processed CSV
if df_global is not None:
    df_global['Category'] = df_global['Description'].apply(categorize_transaction)
    try:
        df_global.to_csv('bank_statements_processed.csv', index=False)
        print("Processed CSV file saved as 'bank_statements_processed.csv'")
    except Exception as e:
        print(f"Error saving processed CSV: {e}")

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Financial Analysis Dashboard", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),

    # Date range selector
    html.Div(id='date-range-container', children=[
        html.Label("Select Date Range:", style={'fontWeight': 'bold', 'marginTop': 20, 'marginLeft': 10}),
        dcc.DatePickerRange(
            id='date-picker',
            start_date=df_global['Date'].min() if df_global is not None else None,
            end_date=df_global['Date'].max() if df_global is not None else None,
            display_format='YYYY-MM-DD',
            style={'marginLeft': 10}
        )
    ] if df_global is not None else [html.Div("Error loading data. Please check the CSV file path.")]),

    # Key metrics cards
    html.Div(id='metrics-cards', style={'display': 'flex', 'justifyContent': 'space-around', 'marginTop': 20}),

    # Charts container
    html.Div([
        html.Div([
            dcc.Graph(id='spending-over-time')
        ], style={'width': '100%'}),

        html.Div([
            html.Div([
                dcc.Graph(id='category-breakdown')
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(id='transaction-type-pie')
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
        ]),

        html.Div([
            html.Div([
                dcc.Graph(id='category-pie')
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(id='debit-distribution')
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
        ]),

        # Table section with category filter
        html.Div([
            html.H3("Top 10 Debits", style={'textAlign': 'center', 'marginTop': 30}),
            html.Div([
                html.Div([
                    html.Label("Filter by Category:", style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.Dropdown(
                        id='category-filter',
                        options=[],
                        value='All',
                        style={'width': '300px'}
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
            html.Div(id='top-debits-table', style={'margin': '0 20px'})
        ], style={'marginTop': 30, 'marginBottom': 30})
    ])
])


@app.callback(
    [Output('metrics-cards', 'children'),
     Output('spending-over-time', 'figure'),
     Output('category-breakdown', 'figure'),
     Output('transaction-type-pie', 'figure'),
     Output('category-pie', 'figure'),
     Output('debit-distribution', 'figure')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_dashboard(start_date, end_date):
    if df_global is None:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Error loading CSV file", showarrow=False)
        return [], empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    # Copy the global dataframe
    df = df_global.copy()

    # Filter by date range
    if start_date and end_date:
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        df = df[mask]

    # Filter out transfers for spending calculations
    df_spending = df[df['Category'] != 'Transfer'].copy()

    # Calculate metrics (excluding transfers)
    total_debits = df_spending['Debit'].sum()
    total_credits = df['Credit'].sum()  # Keep all credits
    net_cash_flow = total_credits - df['Debit'].sum()  # Use all debits for cash flow
    avg_transaction = df_spending[df_spending['Debit'] > 0]['Debit'].mean()

    # Create metric cards
    metrics = [
        html.Div([
            html.H4("Total Spent", style={'color': '#7f8c8d'}),
            html.H2(f"${total_debits:,.2f}", style={'color': '#e74c3c'})
        ], style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 5,
                  'width': '22%'}),

        html.Div([
            html.H4("Total Income", style={'color': '#7f8c8d'}),
            html.H2(f"${total_credits:,.2f}", style={'color': '#27ae60'})
        ], style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 5,
                  'width': '22%'}),

        html.Div([
            html.H4("Net Cash Flow", style={'color': '#7f8c8d'}),
            html.H2(f"${net_cash_flow:,.2f}", style={'color': '#27ae60' if net_cash_flow > 0 else '#e74c3c'})
        ], style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 5,
                  'width': '22%'}),

        html.Div([
            html.H4("Avg Transaction", style={'color': '#7f8c8d'}),
            html.H2(f"${avg_transaction:,.2f}", style={'color': '#3498db'})
        ], style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 5,
                  'width': '22%'}),
    ]

    # Spending over time (excluding transfers)
    daily_spending = df_spending.groupby(df_spending['Date'].dt.date).agg({'Debit': 'sum'}).reset_index()
    daily_income = df.groupby(df['Date'].dt.date).agg({'Credit': 'sum'}).reset_index()
    daily_data = daily_spending.merge(daily_income, on='Date', how='outer').fillna(0)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=daily_data['Date'], y=daily_data['Debit'],
                              mode='lines+markers', name='Spending (excl. transfers)', line=dict(color='#e74c3c')))
    fig1.add_trace(go.Scatter(x=daily_data['Date'], y=daily_data['Credit'],
                              mode='lines+markers', name='Income', line=dict(color='#27ae60')))
    fig1.update_layout(title='Daily Spending and Income', xaxis_title='Date', yaxis_title='Amount ($)',
                       hovermode='x unified')

    # Category breakdown (excluding transfers)
    category_spending = df_spending[df_spending['Debit'] > 0].groupby('Category')['Debit'].sum().sort_values(
        ascending=True)
    fig2 = px.bar(x=category_spending.values, y=category_spending.index, orientation='h',
                  title='Spending by Category', labels={'x': 'Amount ($)', 'y': 'Category'},
                  color=category_spending.values, color_continuous_scale='Reds')
    fig2.update_layout(showlegend=False)

    # Transaction type pie
    transaction_counts = df['Transaction Type'].value_counts()
    fig3 = px.pie(values=transaction_counts.values, names=transaction_counts.index,
                  title='Transaction Types Distribution')

    # Category spending pie chart (excluding transfers)
    category_spending_pie = df_spending[df_spending['Debit'] > 0].groupby('Category')['Debit'].sum()
    fig4 = px.pie(values=category_spending_pie.values, names=category_spending_pie.index,
                  title='Spending Distribution by Category (excl. transfers)',
                  color_discrete_sequence=px.colors.qualitative.Set3)
    fig4.update_traces(textposition='inside', textinfo='percent+label')

    # Debit distribution ranked chart (excluding transfers)
    debit_data = df_spending[df_spending['Debit'] > 0][['Date', 'Debit', 'Category', 'Description']].copy()
    debit_data = debit_data.sort_values('Debit', ascending=False).reset_index(drop=True)
    debit_data['Rank'] = range(1, len(debit_data) + 1)

    fig5 = go.Figure()

    # Add scatter plot for individual debits
    fig5.add_trace(go.Scatter(
        x=debit_data['Rank'],
        y=debit_data['Debit'],
        mode='markers',
        name='Debits',
        marker=dict(size=6, color=debit_data['Debit'], colorscale='Reds', showscale=True,
                    colorbar=dict(title='Amount ($)')),
        text=debit_data['Description'],
        hovertemplate='<b>Rank:</b> %{x}<br><b>Amount:</b> $%{y:.2f}<br><b>Description:</b> %{text}<extra></extra>'
    ))

    # Add trend line
    if len(debit_data) > 1:
        z = np.polyfit(debit_data['Rank'], debit_data['Debit'], 2)  # 2nd degree polynomial
        p = np.poly1d(z)
        fig5.add_trace(go.Scatter(
            x=debit_data['Rank'],
            y=p(debit_data['Rank']),
            mode='lines',
            name='Trend',
            line=dict(color='blue', width=2, dash='dash')
        ))

    fig5.update_layout(
        title='Debit Distribution: Highest to Lowest (excl. transfers)',
        xaxis_title='Debit Rank',
        yaxis_title='Debit Amount ($)',
        hovermode='closest'
    )

    return metrics, fig1, fig2, fig3, fig4, fig5


@app.callback(
    Output('category-filter', 'options'),
    Input('date-picker', 'start_date')
)
def update_category_dropdown(start_date):
    """Populate category dropdown with available categories"""
    if df_global is None:
        return []

    df = df_global.copy()
    categories = ['All'] + sorted(df['Category'].unique().tolist())

    return [{'label': cat, 'value': cat} for cat in categories]


@app.callback(
    Output('top-debits-table', 'children'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('category-filter', 'value'),
     Input('description-filter', 'value')]
)
def update_debits_table(start_date, end_date, selected_category, description_search):
    """Update the top 10 debits table based on date range, category filter, and description search"""
    if df_global is None:
        return html.Div("No data available")

    df = df_global.copy()

    # Filter by date range
    if start_date and end_date:
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        df = df[mask]

    # Filter by category if not 'All'
    if selected_category and selected_category != 'All':
        df = df[df['Category'] == selected_category]

    # Filter by description search
    if description_search and description_search.strip():
        df = df[df['Description'].str.contains(description_search, case=False, na=False)]

    # Get top 10 debits
    top_debits = df[df['Debit'] > 0].nlargest(10, 'Debit')[
        ['Date', 'Description', 'Category', 'Debit', 'Transaction Type']]

    # Format the Date column
    top_debits['Date'] = top_debits['Date'].dt.strftime('%Y-%m-%d')

    # Format the Debit column
    top_debits['Debit'] = top_debits['Debit'].apply(lambda x: f'${x:,.2f}')

    # Create the table
    table = dash_table.DataTable(
        data=top_debits.to_dict('records'),
        columns=[
            {'name': 'Date', 'id': 'Date'},
            {'name': 'Description', 'id': 'Description'},
            {'name': 'Category', 'id': 'Category'},
            {'name': 'Amount', 'id': 'Debit'},
            {'name': 'Transaction Type', 'id': 'Transaction Type'}
        ],
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
            }
        ]
    )

    return table


if __name__ == '__main__':
    app.run(debug=True, port=8050)