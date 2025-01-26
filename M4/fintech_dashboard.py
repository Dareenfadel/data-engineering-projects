import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

def create_dashboard(filename):
    data = pd.read_csv(filename)
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

    # Ensure the issue_date column is properly parsed
    data['issue_date'] = pd.to_datetime(data['issue_date'], errors='coerce')
    data['issue_month'] = data['issue_date'].dt.month
    data['issue_year'] = data['issue_date'].dt.year

    # Key Metrics
    total_loans = data['loan_amount'].sum()
    avg_loan = data['loan_amount'].mean()
    total_borrowers = data['loan_amount'].count()

    # Figures
    fig1 = px.violin(data, x='letter_grade', y='loan_amount', box=True, points="all",
                     title="Distribution of Loan Amounts Across Different Grades")

    fig2 = px.scatter(data, x='annual_inc', y='loan_amount', color='loan_status',
                      title="Loan Amount vs Annual Income Across States",
                      labels={'annual_inc': 'Annual Income', 'loan_amount': 'Loan Amount'},
                      hover_data=['state', 'loan_status'])

    fig3 = px.line(data, x='issue_month', y='loan_amount', color='issue_year', 
                   title="Loan Issuance Trend Over the Months (Filtered by Year)",
                   labels={'issue_month': 'Month', 'loan_amount': 'Loan Amount'})

    avg_loan_by_state = data.groupby('state')['loan_amount'].mean().reset_index()
    avg_loan_by_state = avg_loan_by_state.sort_values(by='loan_amount', ascending=False)

    fig4 = px.bar(
        avg_loan_by_state,
        x='state',
        y='loan_amount',
        title="Average Loan Amount by State",
        labels={'state': 'State', 'loan_amount': 'Average Loan Amount'},
        color='loan_amount',
        color_continuous_scale='Viridis'
    )

    # Compute percentages for loan grades
    grade_counts = data['letter_grade'].value_counts(normalize=True) * 100
    grade_counts = grade_counts.reset_index()
    grade_counts.columns = ['letter_grade', 'percentage']

    fig5_hist = px.bar(
        grade_counts,
        x='letter_grade',
        y='percentage',
        title="Percentage Distribution of Loan Grades",
        labels={'letter_grade': 'Loan Grade', 'percentage': 'Percentage'},
        color_discrete_sequence=['#636EFA']
    )

    # Layout
    app.layout = dbc.Container([

        # User Information
        dbc.Row([ 
            dbc.Col(html.Div([ 
                html.H1("Fintech Dashboard", className="text-center mb-4"),
                html.H5(f"Developed by: Darin Mohamed Abdelaal (ID: 52-21362)", 
                        className="text-center text-muted", style={"marginBottom": "20px"})
            ]), width=12)
        ], style={"backgroundColor": "#f8f9fa", "padding": "20px"}),  # Header Background

        # Summary Cards
        dbc.Row([ 
            dbc.Col(dbc.Card([ 
                dbc.CardHeader("Total Loans Issued"),
                dbc.CardBody(f"${total_loans:,.2f}", className="text-center text-primary")
            ], color="light", inverse=False), width=4), 
            dbc.Col(dbc.Card([ 
                dbc.CardHeader("Average Loan Amount"),
                dbc.CardBody(f"${avg_loan:,.2f}", className="text-center text-success")
            ], color="light", inverse=False), width=4),
            dbc.Col(dbc.Card([ 
                dbc.CardHeader("Total Borrowers"),
                dbc.CardBody(f"{total_borrowers:,}", className="text-center text-info")
            ], color="light", inverse=False), width=4),
        ], className="mb-4"),

        # Loan Amount Distribution Across Grades
        dbc.Row([ 
            dbc.Col([ 
                html.H4("Loan Amount Distribution Across Grades"),
                dcc.Graph(figure=fig1)
            ], width=6), 
            dbc.Col([ 
                html.H4("States with Highest Average Loan Amount"),
                dcc.Graph(figure=fig4)
            ], width=6),
        ], style={"backgroundColor": "#e9ecef", "padding": "20px"}),  # Section Background

        # Loan Amount vs Annual Income
        dbc.Row([ 
            dbc.Col([ 
                html.H4("Loan Amount vs Annual Income Across States"),
                dcc.Dropdown(
                    id='state-dropdown',
                    options=[{'label': state, 'value': state} for state in data['state'].dropna().unique()] + [{'label': 'All States', 'value': 'all'}],
                    value='all',
                    style={'width': '50%'}
                ),
                dcc.Graph(id='loan-amount-vs-annual-income', figure=fig2)
            ], width=12),
        ], className="mb-4"),

        # Loan Issuance Trend
        dbc.Row([ 
            dbc.Col([ 
                html.H4("Loan Issuance Trend by Month"),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': str(year), 'value': year} for year in data['issue_year'].dropna().unique()],
                    value=data['issue_year'].min(),
                    style={'width': '50%'}
                ),
                dcc.Graph(id='loan-issuance-trend', figure=fig3)
            ], width=12),
        ], style={"backgroundColor": "#f8f9fa", "padding": "20px"}),  # Section Background

        # Loan Grades Percentage
        dbc.Row([ 
            dbc.Col([ 
                html.H4("Percentage Distribution of Loan Grades"),
                dcc.Graph(figure=fig5_hist)
            ], width=12)
        ])
    ], fluid=True)

    @app.callback(
        Output('loan-amount-vs-annual-income', 'figure'),
        [Input('state-dropdown', 'value')]
    )
    def update_scatter(selected_state):
        filtered = data if selected_state == 'all' else data[data['state'] == selected_state]
        return px.scatter(filtered, x='annual_inc', y='loan_amount', color='loan_status',
                          title="Loan Amount vs Annual Income Across States",
                          labels={'annual_inc': 'Annual Income', 'loan_amount': 'Loan Amount'},
                          hover_data=['state', 'loan_status'])

    @app.callback(
    Output('loan-issuance-trend', 'figure'),
    [Input('year-dropdown', 'value')]
    )
    def update_trend(selected_year):
        filtered = data[data['issue_year'] == selected_year]

    # Group by month and aggregate loan count and total loan amount
        trend_data = filtered.groupby('issue_month').agg(
        Loan_Count=('loan_amount', 'count'),   # Count of loans
        Loan_Total=('loan_amount', 'sum')     # Total loan amount
    ).reset_index()

    # Create line graph for loan count (or switch to 'Loan_Total')
        fig = px.line(
        trend_data,
        x='issue_month',
        y='Loan_Count',  # Change to 'Loan_Total' for total loan amount
        title=f"Trend of Loan Issuance Over Months ({selected_year})",
        labels={'issue_month': 'Month', 'Loan_Count': 'Number of Loans'},
        markers=True
        )

    # Optional: Ensure x-axis is treated as linear
        fig.update_layout(xaxis=dict(tickmode='linear'))

        return fig


    # Run the Dash server
    app.run_server(host='0.0.0.0', port=8050, debug=False, threaded=True)
