import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson,skellam
EPL=pd.read_csv("https://www.football-data.co.uk/mmz4281/2324/E0.csv")
EPL=EPL[['HomeTeam','AwayTeam','FTHG','FTAG']]
EPL=EPL.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
EPL.head()
EPL = EPL[:-10]
EPL.mean()


goal_model_data = pd.concat([EPL[['HomeTeam','AwayTeam','HomeGoals']].assign(home=1).rename(
            columns={'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}),
           EPL[['AwayTeam','HomeTeam','AwayGoals']].assign(home=0).rename(
            columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})])
poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data,
                        family=sm.families.Poisson()).fit()
poisson_model.summary()

poisson_model.predict(pd.DataFrame(data={'team': 'West Ham', 'opponent': 'Bournemouth',
                                       'home':1},index=[1]))
                                       
poisson_model.predict(pd.DataFrame(data={'team': 'Bournemouth','opponent':'West Ham','home':0},index=[1]))

def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam,
                                                            'opponent': awayTeam,'home':1},
                                                      index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam,
                                                            'opponent': homeTeam,'home':0},
                                                      index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))
simulate_match(poisson_model, 'Fulham', 'Everton',max_goals=1)

Pred= simulate_match(poisson_model, "West Ham", "Bournemouth", max_goals=10)
# home win
np.sum(np.tril(Pred, -1))

#draw
np.sum(np.diag(Pred))

#away win
np.sum(np.triu(Pred, 1))