#!/usr/bin/python3

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import statistics as stat
import json


class SEIRD:
    def __init__(self, filename, population, outfile):
        self.filename = filename
        self.N = population
        self.outfile = outfile
        self.REQUIRED_SAMPLES = 18

    def feed_data(self):
        # try :
        df = pd.read_csv(self.filename, header=0)  # dataset
        # except :
        # print("----")
        # return False
        start_date = df['Date'].iloc[0]
        confirmed = df['Confirmed'].values.tolist()
        recovered = df['Recovered'].values.tolist()
        deaths = df['Deaths'].values.tolist()
        # infected = df['Infected'].values.tolist()
        # confirmed = [ i + r + d for (i, r, d) in zip(infected, recovered, deaths)]
        infected = [c - r - d for (c, r, d)
                    in zip(confirmed, recovered, deaths)]
        # recovered = [ c - i - d for (c, i, d) in zip(confirmed, infected, deaths)]
        available = len(df.index)
        # print("Available", available,"days' data starting from", start_date)
        if available <= self.REQUIRED_SAMPLES:
            return None

        return start_date, available, infected, recovered, deaths

    # Start date - 24th March - India Lockdown
    def create_dates(self, start_date, t_max):
        date = np.array(start_date, dtype=np.datetime64)
        dates = date + np.arange(t_max + 1)
        str_dates = []
        for i in dates:
            str_dates.append(str(i))

        return str_dates

    def seird_model(self, N, str_dates, init_vals, params, t):
        DATE_0, S_0, E_0, I_0, R_0, D_0 = init_vals
        DATE, S, E, I, R, D = [DATE_0], [S_0], [E_0], [I_0], [R_0], [D_0]
        for tic in t[1:]:
            DATE.append(str_dates[tic])
            # print(tic, DATE)
            alpha = params[0]
            beta = params[1]
            gamma = params[2]
            delta = params[3]

            next_S = S[-1] - (beta * (S[-1] / N) * I[-1])  # Susceptible
            # Exposed (Undetected + asymptomatic)
            next_E = E[-1] + (beta * (S[-1] / N) * I[-1]) - (alpha * E[-1])
            next_I = I[-1] + (alpha * E[-1]) - (gamma * I[-1]
                                                ) - (delta * I[-1])  # Hospitalized
            next_R = R[-1] + (gamma * I[-1])  # Recovered
            next_D = D[-1] + (delta * I[-1])  # Deaths
            S.append(next_S)
            E.append(next_E)
            I.append(next_I)
            R.append(next_R)
            D.append(next_D)
        return np.stack([DATE, S, E, I, R, D]).T

    # Spliting the data for one day rolling window approach

    def split_data(self, pred_start, infected, recovered, deaths):
        train_min = 1

        train_max = pred_start + 1
        inf_train = []
        rec_train = []
        death_train = []
        for i in range(train_min + 1, train_max):
            j = i - 2
            for k in range(j, i):
                if infected[k] == 0:
                    infected[k] += 1
            inf_train.append(infected[j:i])
            rec_train.append(recovered[j:i])
            death_train.append(deaths[j:i])
        del i
        del j
        return inf_train, rec_train, death_train

    def param_estimator(self, N, str_dates, inf_train, rec_train, death_train):
        t2 = np.arange(0, 3, 1)
        # print("t2", t2)
        last5_vals = []
        last5_params = []
        t_incub = 5  # Assumption 5 days

        for sample in range(len(inf_train)):
            i_0 = inf_train[sample][0]
            i_1 = inf_train[sample][1]

            i_diff = i_1 - i_0

            r_0 = rec_train[sample][0]
            r_1 = rec_train[sample][1]
            r_diff = r_1 - r_0

            d_0 = death_train[sample][0]
            d_1 = death_train[sample][1]
            d_diff = d_1 - d_0

            # print("train", str_dates[sample],i_0, r_0, d_0)

            if (sample == 0):
                inf_train[sample] = list(map(float, inf_train[sample]))
                g_0 = (rec_train[sample][1] - rec_train[sample]
                [0]) / inf_train[sample][0]
                delta_0 = (
                                  death_train[sample][1] - death_train[sample][0]) / inf_train[sample][0]
                e_0 = t_incub * \
                      (inf_train[sample][1] -
                       ((1 - g_0 - delta_0) * inf_train[sample][0]))

                if e_0 < 1:
                    e_0 = 1

                g_1 = (rec_train[sample + 1][1] -
                       rec_train[sample + 1][0]) / inf_train[sample + 1][0]
                delta_1 = (
                                  death_train[sample + 1][1] - death_train[sample + 1][0]) / inf_train[sample + 1][0]
                e_1 = e_0 + \
                      (t_incub) * (inf_train[sample + 1][1] -
                                   ((1 - g_1 - delta_1) * inf_train[sample + 1][0]))

                e_diff = e_1 - e_0
                s_0 = N - e_0 - i_0 - r_0 - d_0

            a = ((i_diff + r_diff + d_diff) / e_0)  # alpha
            b = ((e_diff + i_diff + r_diff + d_diff)
                 * N) / (s_0 * i_0)  # beta

            g = r_diff / i_0  # gamma
            d = d_diff / i_0  # delta

            init_vals = str_dates[sample], s_0, e_0, i_0, r_0, d_0
            params = a, b, g, d
            # print(params)
            str_dates2 = self.create_dates(str_dates[sample], len(t2))
            pred = self.seird_model(N, str_dates2, init_vals, params, t2)
            # print(pred)
            s_0 = float(pred[1][1])
            e_0 = float(pred[1][2])
            e_1 = float(pred[2][2])
            e_diff = e_1 - e_0

            if ((len(inf_train) - sample) <= 6):
                # print(str_dates[sample])
                last5_vals.append(init_vals)
                last5_params.append([a, b, g, d])
        return last5_params, last5_vals

    def predictions(self, last5_params, last5_vals):
        t7 = np.arange(0, 28, 1)
        preds = []
        for z in range(1, len(last5_params)):  # 4params
            str_dates3 = []
            str_dates3 = self.create_dates(last5_vals[z - 1][0], len(t7))
            results = self.seird_model(
                self.N, str_dates3, last5_vals[z - 1], last5_params[z - 1], t7)
            # t_z = np.arange(7 - z, 14 - z, 1)
            t_z = np.arange(0, 29 - z, 1)
            temp = []
            for day in t_z:
                temp.append([results[day][0], int(float(results[day][3])), int(
                    float(results[day][4])), int(float(results[day][5]))])
                # print(temp)
            preds.append(temp)
            # print("z", z)
        return preds

    def modified_predictions(self, last5_params, recent_vals):
        t21 = np.arange(0, 23, 1)
        preds = []
        for z in range(1, len(last5_params)):  # 5params
            str_dates3 = []
            str_dates3 = self.create_dates(recent_vals[0], len(t21))
            results = self.seird_model(self.N, str_dates3, recent_vals, last5_params[z - 1], t21)
            # t_z = np.arange(7 - z, 14 - z, 1)
            # t_z = np.arange(0, 15 - z, 1)
            temp = []
            for day in range(1, len(t21)):
                temp.append([results[day][0], int(float(results[day][3])), int(float(results[day][4])),
                             int(float(results[day][5]))])
                # print(temp)
            preds.append(temp)
            # print("z", z)
        return preds

    def MAPE(self, actual, predicted):

        mape = 0
        # total_true = 0
        samples = len(actual)
        for i in range(samples):
            true = actual[i]
            # total_true += true
            pred = predicted[i]

            if (true == 0):
                mape_s = 0
            else:
                mape_s = (abs(true - pred) / true)
                # mape_s = abs(true - pred)  #mae
                # print("mape_s", mape_s)

            mape += mape_s

        mape = mape * 100

        return mape / samples

    def param_selection(self):

        infected = []
        recovered = []
        deaths = []
        start_date, available, infected, recovered, deaths = self.feed_data()

        # print(infected)
        # print(deaths)

        pred_start = available
        # pred_start = available - 7
        # print(pred_start)

        inf_train = []
        rec_train = []
        death_train = []

        inf_train, rec_train, death_train = self.split_data(
            pred_start, infected, recovered, deaths)
        # print(rec_train)
        # print(death_train)

        last5_params = []
        last5_vals = []

        str_dates = self.create_dates(start_date, available)
        # print(str_dates)

        last5_params, last5_vals = self.param_estimator(
            self.N, str_dates, inf_train, rec_train, death_train)
        # print("params", last5_params)
        # print("vals", last5_vals)

        recent_vals = last5_vals[-1]

        preds = []
        preds = self.modified_predictions(last5_params, recent_vals)

        # print(preds)
        # print(len(preds))

        v = pred_start - 7

        mape_list = []
        for x in range(len(last5_params) - 1):
            avg_mape = 0
            actual_i = []
            predicted_i = []
            mape_i = 0

            actual_d = []
            predicted_d = []
            mape_d = 0

            actual_r = []
            predicted_r = []
            mape_r = 0

            # print(preds[x][2][0], preds[x][2][1])
            # print(str_dates[v+x+2], infected[v+x+2])
            actual_i.append(infected[v + x + 2])
            predicted_i.append(preds[x][2][1])

            actual_d.append(deaths[v + x + 2])
            predicted_d.append(preds[x][2][3])

            actual_r.append(recovered[v + x + 2])
            predicted_r.append(preds[x][2][2])

            # print("actual i ", actual_i)
            # print("predicted i ", predicted_i)
            mape_i = self.MAPE(actual_i, predicted_i)

            # print("actual d ", actual_d)
            # print("predicted d ", predicted_d)
            mape_d = self.MAPE(actual_d, predicted_d)

            # print("actual r ", actual_r)
            # print("predicted r ", predicted_r)
            mape_r = self.MAPE(actual_r, predicted_r)

            # print("x", x, "mape_i", mape_i, "mape_d", mape_d, "mape_r", mape_r)
            avg_mape = np.around((mape_i + mape_d + mape_r) / 3, decimals=3)
            # print("total mape", avg_mape)
            mape_list.append(avg_mape)
            # mape_list.append(mape_d)
            # print("----------------------------------")
        # print(mape_list)

        best_param = mape_list.index(min(mape_list))
        # print(best)

        dates_p = []
        inf_predicted = []
        rec_predicted = []
        death_predicted = []

        for t in range(21):
            dates_p.append(preds[best_param][len(preds[best_param]) - 21:][t][0])
            inf_predicted.append(
                preds[best_param][len(preds[best_param]) - 21:][t][1])
            rec_predicted.append(
                preds[best_param][len(preds[best_param]) - 21:][t][2])
            death_predicted.append(
                preds[best_param][len(preds[best_param]) - 21:][t][3])

        con_predicted = []

        con_predicted = [i + r + d for i, r,
                                       d in zip(inf_predicted, rec_predicted, death_predicted)]

        final_preds = []

        final_preds.append(dates_p)
        final_preds.append(con_predicted)
        final_preds.append(death_predicted)

        return final_preds

    def save_predictions(self, out):
        jsonOutput = dict()
        predictions = out.values.tolist()

        predictionsObjectsList = list()
        weekObjectsList = list()
        weekObject = dict()
        weekTotalObject = {
            'confirmed': 0,
            'deaths': 0
        }
        week = 1
        day = 0
        for pred in predictions:
            predictionsObject = dict()
            predictionsObject['date'] = pred[0]
            predictionsObject['confirmed'] = pred[1]
            predictionsObject['deaths'] = pred[2]
            weekTotalObject['confirmed'] = predictionsObject['confirmed']
            weekTotalObject['deaths'] = predictionsObject['deaths']
            predictionsObjectsList.append(predictionsObject)
            day += 1
            if day % 7 == 0:
                weekObject[f'Week-{week}'] = {'predictions': []}
                weekObject[f'Week-{week}']['predictions'] = predictionsObjectsList
                weekObject[f'Week-{week}']['total'] = weekTotalObject

                weekTotalObject = {
                    'confirmed': 0,
                    'deaths': 0
                }
                weekObjectsList.append(weekObject)
                weekObject = dict()
                predictionsObjectsList = list()
                week += 1

        jsonOutput["overallPredictions"] = weekObjectsList

        outFileHandler = open(self.outfile + "_projections.json", 'w')
        json.dump(jsonOutput, outFileHandler)

    def final_run(self):
        # final_i = []
        # final_d = []
        preds = []
        final_preds = self.param_selection()

        out = pd.DataFrame(final_preds, index=[
            "Date", "Confirmed", "Deceased"])

        out = out.T

        print(out)
        self.save_predictions(out)

        return out

#
# infile = './Datasets/Russia/Moscow.csv'
# # infile = 'Thane.csv'
#
# N = 12537954
# # # N = 9426959  # pune
# # N = 11060148 #thane
# model = SEIRD(infile, N, "hihi")
#
# output = model.final_run()
# print(output)
