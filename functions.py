import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pyspark as ps
import pyspark.sql.functions as f
from pyspark.sql.functions import col
from math import exp
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, FloatType, IntegerType





def cpiLookup(year):
    '''Given a year returns the CPI for that year, needs to be functionalized for spark udf
    Parameters
    ----------
    year : float/int/string will be casted to a float with ####.0 for cpi_dict lookup
    
    Returns
    -------
    the CPI for that year
    '''
    #cpi from online spreadsheet
    cpi_list = [[1926,17.7],[1927,17.4],[1928,17.2],[1929,17.2],[1930, 16.7],[1931,15.2],[1932,13.6],[1933,12.9],[1934,13.4], [1935,13.7],[1936,13.9],[1937,14.4], [1938,14.1],[1939,13.9],[1940,14.],[1941,14.7],
 [1942,16.3],[1943,17.3],[1944,17.6],[1945,18.],[1946,19.5],[1947,22.3],[1948,24.],[1949,23.8],[1950,24.1],[1951,26.],[1952,26.6],[1953,26.8],[1954,26.9],[1955,26.8],[1956, 27.2],
 [1957,28.1],[1958,28.9],[1959,29.2],[1960,29.6],[1961,29.9],[1962,30.3],[1963,30.6],[1964,31.],[1965,31.5],[1966,32.5],[1967,33.4],[1968,34.8],[1969,36.7],[1970,38.8],
 [1971,40.5],[1972,41.8],[1973,44.4],[1974,49.3],[1975,53.8],[1976,56.9],[1977,60.6],[1978,65.2],[1979,72.6],[1980,82.4],[1981,90.9],[1982,96.5],[1983,99.6],[1984,103.9],
 [1985,107.6],[1986,109.6],[1987,113.6],[1988,118.3],[1989,124.],[1990,130.7],[1991,136.2],[1992,140.3],[1993,144.5],[1994,148.2],[1995,152.4],[1996,156.9],[1997,160.5],
 [1998,163.],[1999,166.6],[2000,172.2],[2001,177.1],[2002,179.9],[2003,184.],[2004,188.9],[2005,195.3],[2006,201.6],[2007,207.3],[2008,215.3],[2009,214.5],[2010,218.1],
 [2011,224.9],[2012,229.6],[2013,233.],[2014,236.7],[2015,237.],[2016,240.],[2017,245.1],[2018,250.5],[2019,257.],[2020,260.5]]

    cpi_dict = {}
    for a in cpi_list:
        cpi_dict[a[0]] = a[1]
    
    year = int(year)
    return cpi_dict[year]



def cpiAdjustedVal(cpi, value):
    '''given a cpi and a value converts the value to 2020 dollars
    ----------
    cpi : the consumer price index for the year of the value
    value : a value in dollars in its year to adjust
    
    
    Returns
    -------
    the cpi adjusted value
    '''
    return value*260.5/cpi


def compare_by_net(df_1, df_2, label_1, label_2, alpha=.05, withPlot = True):
    '''
    runs a mann-whitney-u test to see which is more successful
        
    PARAMETERS
    ----------
    df_1 (pandas_df): one of the dataFrames to compare
    
    df_2 (pandas_df): the other dataFrame to compare
    
    alpha (float): rejection threshold
    
    withPlot (bool): to plot or not to plot the normal distributions of the data
    
    RETURNS
    -------
    histogram of both dataFrames
    list : p-value for mann-whitney 
    '''
    
    plt.hist(df_1.net, bins = 15, label = label_1, alpha = .5)
    plt.hist(df_2.net, bins = 15, label = label_2, alpha = .5)
    plt.legend()
    
    u_stat, p_utest = stats.mannwhitneyu(df_1.net, df_2.net, alternative = 'greater')
    return p_utest



def compare_genre_net(data_frame, genre1:str, genre2:str, start_year = 1930, end_year = 2020, graph_title = 0):
    '''
    creates bootstrap samples of two the films of two genres, compares them and graphs their distributions and the distribution
        of the differences. Also returns a value for the winner/if it's a tie and the 90% confidence interval of the difference
        
    PARAMETERS
    -----------
    data_frame (spark_df): data_frame from which to collect budgets and gross incomes 
    
    genre1 (str): string of the first genre to compare, formated how IMDb likes i.e. ('Comedy','Drama','Horror')
    
    genre2 (str): string of the second genre to compare, formated how IMDb likes
    
    start_year (int): first year to look at movies from in the data_frame, preset is 1930
    
    end_year (int): last year to look at movies from in the data_frame, preset is 2020
    
    graph_title (str): if str is entered for the graph_title, names the graph by that name, otherwise uses naming convention w/ genre names
    
    RETURNS
    -------
    tuple (winner, [5%, 95%], 50%) where winner is -1 if the second entry performs better, 1 if the first performs better and 0 if inconclusive
    '''
    
    
    
    genre1_df = data_frame.select('adjusted_budget','adjusted_worlwide_gross_income')\
                                  .where((data_frame.genre.rlike(f'{genre1}')) &\
                                         (data_frame.year >= start_year) & (data_frame.year <= end_year))
    genre1_df = genre1_df.withColumn('net', genre1_df.adjusted_worlwide_gross_income - genre1_df.adjusted_budget)
    genre1_pandas = genre1_df.toPandas()

    genre2_df = data_frame.select('adjusted_budget','adjusted_worlwide_gross_income')\
                                  .where((data_frame.genre.rlike(f'{genre2}')) &\
                                         (data_frame.year >= start_year) & (data_frame.year <= end_year))
    genre2_df = genre2_df.withColumn('net', genre2_df.adjusted_worlwide_gross_income - genre2_df.adjusted_budget)
    genre2_pandas = genre2_df.toPandas()
    
    difference = []
    mean_gen1 = []
    mean_gen2 = []
    for _ in range(10000):
        min_movies = min(genre2_pandas.net.count(), genre1_pandas.net.count())
        sample1 = genre1_pandas.net.sample(min_movies, replace = True)
        sample2 = genre2_pandas.net.sample(min_movies, replace = True)
        difference.append(np.mean(sample1)-np.mean(sample2))
        mean_gen1.append(np.mean(sample1))
        mean_gen2.append(np.mean(sample2))
    plt.hist(difference, alpha = .5, label = 'Difference')
    plt.hist(mean_gen1, alpha = .5, label = f'{genre1}')
    plt.hist(mean_gen2, alpha = .5, label = f'{genre2}')
    plt.legend()
    if graph_title != 0:
        plt.title(f'{graph_title}')
        plt.savefig(f'images/{graph_title}.png')
    else:
        plt.title(f'Bootstrapped Simulation of {genre1} and {genre2}')
        plt.savefig(f'images/{genre1}_{genre2}_net.png')
    left_endpoint = np.percentile(difference, 10)
    right_endpoint = np.percentile(difference, 90)
    
    if (left_endpoint < 0) & (right_endpoint <0):
        return (-1, [left_endpoint, right_endpoint], np.percentile(difference, 50))
    elif (left_endpoint > 0) & (right_endpoint > 0):
        return (1, [left_endpoint, right_endpoint], np.percentile(difference, 50))
    else:
        return (0, [left_endpoint, right_endpoint], np.percentile(difference, 50))



def compare_studios_net(data_frame, studio1:str, studio2:str, start_year = 1930, end_year = 2020):

    studio1_df = data_frame.select('adjusted_budget','adjusted_worlwide_gross_income')\
                                  .where((data_frame.production_company.rlike(f'{studio1}')) &\
                                         (data_frame.year >= start_year) & (data_frame.year <= end_year))
    studio1_df = studio1_df.withColumn('net', studio1_df.adjusted_worlwide_gross_income - studio1_df.adjusted_budget)
    studio1_pandas = studio1_df.toPandas()

    studio2_df = data_frame.select('adjusted_budget','adjusted_worlwide_gross_income')\
                                  .where((data_frame.production_company.rlike(f'{studio2}')) &\
                                         (data_frame.year >= start_year) & (data_frame.year <= end_year))
    studio2_df = studio2_df.withColumn('net', studio2_df.adjusted_worlwide_gross_income - studio2_df.adjusted_budget)
    studio2_pandas = studio2_df.toPandas()
    
    difference = []
    mean_stu1 = []
    mean_stu2 = []
    for _ in range(10000):
        min_movies = min(studio2_pandas.net.count(), studio1_pandas.net.count())
        sample1 = studio1_pandas.net.sample(min_movies, replace = True)
        sample2 = studio2_pandas.net.sample(min_movies, replace = True)
        difference.append(np.mean(sample1)-np.mean(sample2))
        mean_stu1.append(np.mean(sample1))
        mean_stu2.append(np.mean(sample2))
    plt.hist(difference, alpha = .5, label = 'Difference')
    plt.hist(mean_stu1, alpha = .5, label = f'{studio1}')
    plt.hist(mean_stu2, alpha = .5, label = f'{studio2}')
    plt.legend()
    plt.title(f'Bootstrapped Simulation of {studio1} and {studio2}')
    plt.savefig(f'images/{studio1}_{studio2}_net.png')
    left_endpoint = np.percentile(difference, 10)
    right_endpoint = np.percentile(difference, 90)
    
    if (left_endpoint < 0) & (right_endpoint <0):
        return (-1, [left_endpoint, right_endpoint], np.percentile(difference, 50))
    elif (left_endpoint > 0) & (right_endpoint > 0):
        return (1, [left_endpoint, right_endpoint], np.percentile(difference, 50))
    else:
        return (0, [left_endpoint, right_endpoint], np.percentile(difference, 50))


def compare_actors_net(data_frame, actor1:str, actor2:str, ax,color =['b','r','g'], start_year = 1930, end_year = 2020, plot = True):

    actor1_df = data_frame.select('adjusted_budget','adjusted_worlwide_gross_income')\
                                  .where((data_frame.actors.rlike(f'{actor1}')) &\
                                         (data_frame.year >= start_year) & (data_frame.year <= end_year))
    actor1_df = actor1_df.withColumn('net', actor1_df.adjusted_worlwide_gross_income - actor1_df.adjusted_budget)
    actor1_pandas = actor1_df.toPandas()

    actor2_df = data_frame.select('adjusted_budget','adjusted_worlwide_gross_income')\
                                  .where((data_frame.actors.rlike(f'{actor2}')) &\
                                         (data_frame.year >= start_year) & (data_frame.year <= end_year))
    actor2_df = actor2_df.withColumn('net', actor2_df.adjusted_worlwide_gross_income - actor2_df.adjusted_budget)
    actor2_pandas = actor2_df.toPandas()
    
    difference = []
    mean_act1 = []
    mean_act2 = []
    for _ in range(10000):
        min_movies = min(actor2_pandas.net.count(), actor1_pandas.net.count())
        sample1 = actor1_pandas.net.sample(min_movies, replace = True)
        sample2 = actor2_pandas.net.sample(min_movies, replace = True)
        difference.append(np.mean(sample1)-np.mean(sample2))
        mean_act1.append(np.mean(sample1))
        mean_act2.append(np.mean(sample2))
    left_endpoint = np.percentile(difference, 5)
    right_endpoint = np.percentile(difference, 95)
    if plot:
        ax[0].hist(difference, alpha = .5, color = color[0], label = 'Difference', bins = 25)
        ax[0].axvline(left_endpoint, color = color[1])
        ax[0].axvline(right_endpoint, color = color[1], label = '90% Confidence Interval')
        ax[0].set_xlabel('2020 US Dollars (100 Millions)')
        ax[1].hist(mean_act1, alpha = .5, color = color[1], label = f'{actor1}', bins = 25)
        ax[1].hist(mean_act2, alpha = .5, color = color[2], label = f'{actor2}', bins = 25)
        ax[1].set_xlabel('2020 US Dollars (100 Millions)')
        ax[0].legend()
        ax[1].legend()
        ax[0].set_title(f'Difference Between Career Simulations of {actor1} and {actor2}')
        ax[1].set_title(f'Bootstrapped Career Simulation of {actor1} and {actor2}')
        plt.tight_layout()
        plt.savefig(f'images/{actor1}_{actor2}_net.png')
    
    if (left_endpoint < 0) & (right_endpoint <0):
        return (-1, [left_endpoint, right_endpoint], np.percentile(difference, 50))
    elif (left_endpoint > 0) & (right_endpoint > 0):
        return (1, [left_endpoint, right_endpoint], np.percentile(difference, 50))
    else:
        return (0, [left_endpoint, right_endpoint], np.percentile(difference, 50))

def compare_actors_roi(data_frame, actor1:str, actor2:str, ax,color = ['b','r','g'], start_year = 1930, end_year = 2020,plot = True):

    actor1_df = data_frame.select('adjusted_budget','adjusted_worlwide_gross_income')\
                                  .where((data_frame.actors.rlike(f'{actor1}')) &\
                                         (data_frame.year >= start_year) & (data_frame.year <= end_year))
    actor1_df = actor1_df.withColumn('roi', (actor1_df.adjusted_worlwide_gross_income-actor1_df.adjusted_budget)/actor1_df.adjusted_budget)
    actor1_pandas = actor1_df.toPandas()

    actor2_df = data_frame.select('adjusted_budget','adjusted_worlwide_gross_income')\
                                  .where((data_frame.actors.rlike(f'{actor2}')) &\
                                         (data_frame.year >= start_year) & (data_frame.year <= end_year))
    actor2_df = actor2_df.withColumn('roi', (actor2_df.adjusted_worlwide_gross_income-actor2_df.adjusted_budget)/actor2_df.adjusted_budget)
    actor2_pandas = actor2_df.toPandas()
    
    difference = []
    mean_act1 = []
    mean_act2 = []
    for _ in range(10000):
        min_movies = min(actor2_pandas.roi.count(), actor1_pandas.roi.count())
        sample1 = actor1_pandas.roi.sample(min_movies, replace = True)
        sample2 = actor2_pandas.roi.sample(min_movies, replace = True)
        difference.append(np.mean(sample1)-np.mean(sample2))
        mean_act1.append(np.mean(sample1))
        mean_act2.append(np.mean(sample2))
    left_endpoint = np.percentile(difference, 5)
    right_endpoint = np.percentile(difference, 95)
    if plot:
        ax[0].hist(difference, alpha = .5, label = 'Difference', bins = 25, color = color[0])
        ax[0].axvline(left_endpoint, color = color[1])
        ax[0].axvline(right_endpoint, color = color[1], label = '90% Confidence Interval')
        ax[1].hist(mean_act1, alpha = .5, label = f'{actor1}', bins = 25, color = color[1])
        ax[1].hist(mean_act2, alpha = .5, label = f'{actor2}', bins = 25, color = color[2])
        ax[0].legend()
        ax[1].legend()
        ax[0].set_title(f'Difference Between Career Simulations of {actor1} and {actor2}')
        ax[1].set_title(f'Bootstrapped Simulation of {actor1} and {actor2}')
        plt.tight_layout()
        plt.savefig(f'images/{actor1}_{actor2}_roi.png')
    
    if (left_endpoint < 0) & (right_endpoint <0):
        return (-1, [left_endpoint, right_endpoint], np.percentile(difference, 50))
    elif (left_endpoint > 0) & (right_endpoint > 0):
        return (1, [left_endpoint, right_endpoint], np.percentile(difference, 50))
    else:
        return (0, [left_endpoint, right_endpoint], np.percentile(difference, 50))
    
    
def compare_genre_roi(data_frame, genre1:str, genre2:str,graph_title=0, start_year = 1930, end_year = 2020):

    genre1_df = data_frame.select('adjusted_budget','adjusted_worlwide_gross_income')\
                                  .where((data_frame.genre.rlike(f'{genre1}')) &\
                                         (data_frame.year >= start_year) & (data_frame.year <= end_year))
    genre1_df = genre1_df.withColumn('roi', (genre1_df.adjusted_worlwide_gross_income-genre1_df.adjusted_budget)/genre1_df.adjusted_budget)
    genre1_pandas = genre1_df.toPandas()

    genre2_df = data_frame.select('adjusted_budget','adjusted_worlwide_gross_income')\
                                  .where((data_frame.genre.rlike(f'{genre2}')) &\
                                         (data_frame.year >= start_year) & (data_frame.year <= end_year))
    genre2_df = genre2_df.withColumn('roi', (genre2_df.adjusted_worlwide_gross_income-genre2_df.adjusted_budget)/genre2_df.adjusted_budget)
    genre2_pandas = genre2_df.toPandas()
    
    difference = []
    mean_gen1 = []
    mean_gen2 = []
    for _ in range(10000):
        min_movies = min(genre2_pandas.roi.count(), genre1_pandas.roi.count())
        sample1 = genre1_pandas.roi.sample(min_movies, replace = True)
        sample2 = genre2_pandas.roi.sample(min_movies, replace = True)
        difference.append(np.mean(sample1)-np.mean(sample2))
        mean_gen1.append(np.mean(sample1))
        mean_gen2.append(np.mean(sample2))
    plt.hist(difference, alpha = .5, label = 'Difference')
    plt.hist(mean_gen1, alpha = .5, label = f'{genre1}')
    plt.hist(mean_gen2, alpha = .5, label = f'{genre2}')
    plt.legend()
    if graph_title != 0:
        plt.title(f'{graph_title}')
    else:
        plt.title(f'Bootstrapped Simulation of {genre1} and {genre2}')
    left_endpoint = np.percentile(difference, 10)
    right_endpoint = np.percentile(difference, 90)
    
    if (left_endpoint < 0) & (right_endpoint <0):
        return (-1, [left_endpoint, right_endpoint])
    elif (left_endpoint > 0) & (right_endpoint > 0):
        return (1, [left_endpoint, right_endpoint])
    else:
        return (0, [left_endpoint, right_endpoint])

    
def compare_genre_budget_class_roi(data_frame, genre: str, axs, genre_colors = ['b','r','y','g'], budget_buckets = [1e6, 1e7, 7.5e7, 2e8], graph_title=0, start_year = 1930, end_year = 2020):
    
    low, mid, high, huge = budget_buckets

    bucket1_df = data_frame.select('adjusted_budget','adjusted_worlwide_gross_income')\
                                  .where((data_frame.genre.rlike(f'{genre}')) &\
                                         (data_frame.adjusted_budget >= low) & (data_frame.adjusted_budget < mid) &\
                                         (data_frame.year >= start_year) & (data_frame.year <= end_year))
    bucket1_df = bucket1_df.withColumn('roi', (bucket1_df.adjusted_worlwide_gross_income - bucket1_df.adjusted_budget)/bucket1_df.adjusted_budget)
    bucket1_pandas = bucket1_df.toPandas()
    
    bucket2_df = data_frame.select('adjusted_budget','adjusted_worlwide_gross_income')\
                                  .where((data_frame.genre.rlike(f'{genre}')) &\
                                         (data_frame.adjusted_budget >= mid) & (data_frame.adjusted_budget < high) &\
                                         (data_frame.year >= start_year) & (data_frame.year <= end_year))
    bucket2_df = bucket2_df.withColumn('roi', (bucket2_df.adjusted_worlwide_gross_income - bucket2_df.adjusted_budget)/bucket2_df.adjusted_budget)
    bucket2_pandas = bucket2_df.toPandas()
    
    bucket3_df = data_frame.select('adjusted_budget','adjusted_worlwide_gross_income')\
                                  .where((data_frame.genre.rlike(f'{genre}')) &\
                                         (data_frame.adjusted_budget >= high) & (data_frame.adjusted_budget < huge) &\
                                         (data_frame.year >= start_year) & (data_frame.year <= end_year))
    bucket3_df = bucket3_df.withColumn('roi', (bucket3_df.adjusted_worlwide_gross_income - bucket3_df.adjusted_budget)/bucket3_df.adjusted_budget)
    bucket3_pandas = bucket3_df.toPandas()
    
    bucket4_df = data_frame.select('adjusted_budget','adjusted_worlwide_gross_income')\
                                  .where((data_frame.genre.rlike(f'{genre}')) &\
                                         (data_frame.adjusted_budget >= huge) &\
                                         (data_frame.year >= start_year) & (data_frame.year <= end_year))
    bucket4_df = bucket4_df.withColumn('roi', (bucket4_df.adjusted_worlwide_gross_income - bucket4_df.adjusted_budget)/bucket4_df.adjusted_budget)
    bucket4_pandas = bucket4_df.toPandas()
    

    mean_buc1 = []
    mean_buc2 = []
    mean_buc3 = []
    mean_buc4 = []
    for _ in range(10000):
        min_movies = min(bucket1_pandas.roi.count(),bucket2_pandas.roi.count(),bucket3_pandas.roi.count(),bucket4_pandas.roi.count())
        sample1 = bucket1_pandas.roi.sample(min_movies, replace = True)
        sample2 = bucket2_pandas.roi.sample(min_movies, replace = True)
        sample3 = bucket3_pandas.roi.sample(min_movies, replace = True)
        sample4 = bucket4_pandas.roi.sample(min_movies, replace = True)
        mean_buc1.append(np.mean(sample1))
        mean_buc2.append(np.mean(sample2))
        mean_buc3.append(np.mean(sample3))
        mean_buc4.append(np.mean(sample4))
    means = [mean_buc1, mean_buc2, mean_buc3, mean_buc4]
    counts = [0,1,2,3]
    for ax, m, count, genre_color in zip(axs.flatten(), means, counts, genre_colors):
        ax.hist(m,alpha = .6, color = genre_color, bins = 25)
        if count != 3:
            ax.set_title(f'{genre}: CPI Adjusted Budget \$ {str(budget_buckets[count])[:-8]} M - \$ {str(budget_buckets[count+1])[:-8]} M')
        else:
            ax.set_title(f'{genre}: CPI Adjusted Budget > $ {str(budget_buckets[count])[:-8]} M')
        if np.percentile(m,90) > 5:
            ax.set_xlim(-1,20)
        else:
            ax.set_xlim(-1,4)
    plt.tight_layout()
    #if graph_title != 0:
        #plt.title(f'{graph_title}')
    #else:
        #plt.title(f'Bootstrapped Simulation of {genre} by Budget Class')
    le_1 = round(np.percentile(mean_buc1, 10),3)
    re_1 = round(np.percentile(mean_buc1, 90),3)
    le_2 = round(np.percentile(mean_buc2, 10),3)
    re_2 = round(np.percentile(mean_buc2, 90),3)
    le_3 = round(np.percentile(mean_buc3, 10),3)
    re_3 = round(np.percentile(mean_buc3, 90),3)
    le_4 = round(np.percentile(mean_buc4, 10),3)
    re_4 = round(np.percentile(mean_buc4, 90),3)
    
    return [[le_1, re_1],[le_2,re_2],[le_3,re_3],[le_4,re_4]]
    
    
    
    
def run_tournament_round(data_frame, tournament_list):
    '''
    tournament_list is paired and even
    '''
    winner_list = []
    for idx in range(0, len(tournament_list),2):
        if tournament_list[idx] == 'Inconclusive':
            winner_list.append(tournament_list[idx+1])
        elif tournament_list[idx+1] == 'Inconclusive':
            winner_list.append(tournament_list[idx])
        else:
            winner_1, result_1 = compare_actors_net(data_frame, tournament_list[idx], tournament_list[idx+1], plot = False)
            if winner_1 == 1:
                winner_list.append(tournament_list[idx])
            elif winner_1 == -1:
                winner_list.append(tournament_list[idx+1])
            else:
                winner_2, result_2 = compare_actors_roi(data_frame, tournament_list[idx], tournament_list[idx+1], plot = False)
                if winner_2 == 1:
                    winner_list.append(tournament_list[idx])
                elif winner_2 == -1:
                    winner_list.append(tournament_list[idx+1])
                elif (result_1 > 0) & (result_2 > 0):
                    winner_list.append(tournament_list[idx])
                elif (result_1 < 0) & (result_2 < 0):
                    winner_list.append(tournament_list[idx+1])
                else:
                    winner_list.append('Inconclusive')
    return winner_list


def round_robin_8(data_frame, rr_list):
    '''
    determine seeds, groups of 8 actors
    '''
    scores = np.array([0,0,0,0,0,0,0,0])
    for idx, actor in enumerate(rr_list):
        if idx != 7:
            for a in range(1,len(rr_list) - 1 - idx):
                winner_1, result_1 = compare_actors_net(data_frame, rr_list[idx], rr_list[idx+a], plot = False)
                if winner_1 == 1:
                    scores[idx] += 5
                elif winner_1 == -1:
                    scores[idx+a] += 5
                else:
                    winner_2, result_2 = compare_actors_roi(data_frame, rr_list[idx], rr_list[idx+a], plot = False)
                    if winner_2 == 1:
                        scores[idx] += 5
                    elif winner_2 == -1:
                        scores[idx+a] += 5
                    elif (result_1 > 0) & (result_2 > 0):
                        scores[idx] += 3
                    elif (result_1 < 0) & (result_2 < 0):
                        scores[idx+a] += 3
    top_2_idx = np.argsort(scores)[-2:]
    top_2_actors = [rr_list[i] for i in top_2_idx]
    return top_2_actors