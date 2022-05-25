import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer


class Kursova:
    data = []
    data_with_target = []
    data_with_target_copy = []

    def save(self):
        self.data.to_csv("winequality-red-fixed.csv", index=False, sep=';')

    def load(self):
        print(f"{'*' * 84}\nЗавантаження файла winequality-red.csv для обробки")
        self.data = pd.read_csv('winequality-red.csv')
        print(self.data.sample(5))
        print(f"{'*' * 84}\nЗаміна назв стовпчиків шляхом заміни пробілів на підчеркування ")
        self.data.columns = [column_name.replace(' ', '_') for column_name in self.data.columns]
        print(self.data.sample(5))
        print(f"{'*' * 84}\nВідбір ознак, визначених умовами завдання. Видалення дублікатів рядків, створення копії")
        selected_features = list(map(lambda x: x - 1, [5, 7, 8, 9, 10, 11]))
        self.data_with_target = self.data.iloc[:, selected_features + [len(self.data.columns) - 1]]
        self.data_with_target = self.data_with_target.drop_duplicates().copy()
        self.data_with_target_copy = self.data_with_target.copy()
        self.data_with_target_copy = self.data_with_target_copy.astype(float).round(5)
        print(self.data_with_target.sample(5))

        print(f"{'*' * 84}\nСтворення копії датасету без цільової змінної ")
        self.data = self.data_with_target.iloc[:, :-1]
        print(self.data.sample(5))

    def investigation(self):
        print(f"{'*' * 84}\nОтримання стандартних статистик для нумеричних ознак датасету ")
        with pd.option_context('display.max_columns', len(self.data_with_target)):
            print(self.data_with_target.describe())
        print(f"{'*' * 84}\nВизначення кількості унікальних значень для кожної з ознак ")
        for col in self.data.columns:
            print(f'{col}: {self.data[col].nunique()}')
        print(f"{'*' * 84}\nВивчення попарної кореляції ознак. Діаграма відображає розподіл значень ознак")
        sns.set()
        g = sns.PairGrid(self.data_with_target)
        g.map_diag(plt.hist, edgecolor="w")
        g.map_offdiag(plt.scatter, edgecolor="w", s=40)
        plt.subplots_adjust(top=0.95)
        plt.suptitle("Попарна кореляція ознак", fontsize=16)
        plt.show()
        sns.histplot(x=self.data_with_target['quality'])
        plt.title('Розподіл цільової змінної', fontsize=20)
        plt.show()
        print(
            f"Спостерігаємо несбалансованість класів. При цьому класи 4 та 8 є малочисельними, а в 3 не вистачить значень (спостережень) для достаньої якості прогнозування.")
        print(f"{'*' * 84}\nРозподіл значень та викидів")
        fig, ax = plt.subplots(ncols=2, nrows=len(self.data.columns), figsize=(12, 18))
        for n, feature in enumerate(self.data.columns):
            sns.boxplot(data=self.data, x=feature, ax=ax[n, 0], showmeans=True)
            sns.violinplot(data=self.data, x=feature, ax=ax[n, 1])
        plt.suptitle('Розподіл значень та викидів')
        plt.subplots_adjust(top=0.95)
        plt.show()
        print(f"{'*' * 84}\nМатриця кореляції ознак")
        sns.set(style="white")
        corr = self.data.corr()
        corr = corr.apply(np.abs)
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        mask_annot = np.absolute(corr.values) >= 0.50
        annot_arr = np.where(mask_annot, corr.values.round(2), np.full((6, 6), ""))
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, ax=ax, annot=annot_arr, fmt="s")
        plt.subplots_adjust(top=0.95)
        plt.suptitle("Матриця кореляції", fontsize=20)
        plt.yticks(rotation=0)
        b, t = plt.ylim()
        b += 0.5
        t -= 0.5
        plt.ylim(b, t)
        plt.show()
        print(f"{'*' * 84}\nКореляція (Пірсона): цільова змінна 'quality' з ознаками")
        fig, ax = plt.subplots(figsize=(16, 2))
        data = self.data_with_target.corr().loc[['quality'], :].drop('quality', axis=1)
        sns.heatmap(data=data, cmap='cividis', annot=True, ax=ax)
        ax.set_title(f'Кореляція (Пірсона): цільова змінна quality з ознаками', fontsize=14)
        ax.set_xlabel(f'Ознаки')
        ax.tick_params(axis='x', labelrotation=45)
        plt.show()
        print(
            f"{'*' * 84}\nУсунення мультиколінеарності, усунення ознак, що мають наднизьку кореляцію з цільовою змінною")
        self.data.drop(columns=['total_sulfur_dioxide', 'density'], inplace=True)
        self.data_with_target = self.data_with_target.drop(
            columns=['total_sulfur_dioxide', 'density']).drop_duplicates()
        print(self.data_with_target.sample(5))

    def basic_linear_regression(self, df=None, features=None, target='', X=None, y=None, print_results=True):
        """ """
        if df is not None:
            X = df[features]
            y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        R2 = metrics.r2_score(y_test, y_pred)
        MSE = metrics.mean_squared_error(y_test, y_pred)
        RMSE = np.sqrt(MSE)
        MAE = metrics.mean_absolute_error(y_test, y_pred)

        if print_results:
            print(f'Коефіцієнт детермінації R2: {R2:.4f}\n'
                  f'Cередньоквадратичне відхилення RMSE: {RMSE:.4f}\n'
                  f'Cередня абсолютна похибка MAE: {MAE:.4f}\n'
                  f'Cередньоквадратична похибка MSE: {MSE:.4f}')
        else:
            return R2

    def lof(self):
        df_masks = pd.DataFrame(index=self.data_with_target.index)
        for col in self.data_with_target.columns:
            n = int(self.data_with_target[col].shape[0] * 0.1)
            df_masks[f'mask_{col}'] = LOF(n_neighbors=n).fit_predict(self.data_with_target[col].values.reshape(-1, 1))
        df_masks['final_mask'] = df_masks.sum(axis=1) > 3
        return self.data_with_target.loc[df_masks['final_mask']]

    def basic_logistic_regression(self, df=None, features=None, target='', X=None, y=None):
        """ """
        if df is not None:
            X = df[features]
            y = np.vectorize(lambda y: 0 if y <= 5 else 1)(df[target])
        else:
            y = np.vectorize(lambda y: 0 if y <= 5 else 1)(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=10000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = model.decision_function(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        roc_auc = metrics.roc_auc_score(y_test, y_score)

        precisions, recalls, thresholds_pr = metrics.precision_recall_curve(y_test, y_score)
        metrics.roc_curve(y_test, y_score)

        print(f'Accuracy: {accuracy:.4f}',
              f'Precision: {precision:.4f}',
              f'Recall: {recall:.4f}',
              f'F1: {f1:.4f}',
              f'Площа під кривою (AUC ROC): {roc_auc:.4f}',
              sep='\n')

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
        fig.suptitle('Візуалізація логістичної регресії', fontsize=20)

        metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False, ax=ax[0, 0])
        ax[0, 0].set_title('Матриця помилок')
        ax[0, 1].plot(thresholds_pr, precisions[:-1], 'b--', label='Precision')
        ax[0, 1].plot(thresholds_pr, recalls[:-1], 'g--', label='Recall')
        ax[0, 1].set_title('Precision та Recall')
        metrics.PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax[1, 0])
        ax[1, 0].set_title('Крива Precision-Recall')
        metrics.RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax[1, 1])
        ax[1, 1].set_title('ROC-крива')

        fig.tight_layout()
        plt.show()

    def modeling(self):
        print(f"{'*' * 84}\nЛінійна регресія ")
        self.basic_linear_regression(self.data_with_target, self.data.columns, 'quality')
        print(f"{'*' * 84}\nВидалення викидів з використання моделі LOF (локальний рівень викиду) ")
        current_shape = self.data_with_target.shape[0]
        self.lof()
        print(f'Розмір до видалення викидів: {current_shape}, після видалення: {self.data_with_target.shape[0]}')
        print(f"{'*' * 84}\nПобудова моделі після видалення викидів")
        self.basic_linear_regression(self.data_with_target, self.data.columns, 'quality')
        print(f"{'*' * 84}\nВидалення викидів шляхом обмеження значень цільової змінної та застосування моделі LOF")
        self.data_with_target = self.data_with_target_copy.loc[
            self.data_with_target_copy.quality.between(3.9, 7.1)].copy()
        df_masks = pd.DataFrame(index=self.data_with_target.index)
        for col in self.data_with_target.columns:
            n = int(self.data_with_target[col].shape[0] * 0.1)
            df_masks[f'mask_{col}'] = LOF(n_neighbors=n).fit_predict(self.data_with_target[col].values.reshape(-1, 1))
        df_masks['final_mask'] = df_masks.sum(axis=1) > 3
        self.data_with_target = self.data_with_target.loc[df_masks['final_mask']]
        print(f'Розмір до видалення викидів: {current_shape}, після видалення: {self.data_with_target.shape[0]}')
        print(f"{'*' * 84}\nПобудова моделі після видалення більшої кількості викидів")
        self.basic_linear_regression(self.data_with_target, self.data.columns, 'quality')
        print(
            f"Усунення викидів з використання LOF незначно покращує результати застосування моделі лінійної регресії. "
            f"Усунення викидів шляхом відкидання спостережень з крайовими значеннями цільової змінної погіршує результат відносно базового. "
            f"Усі моделі дозволяють пояснити лише до 30% варіативності цільової змінної, що не може вважатися хорошим результатом. "
            f"Задача потребує застосування більш складних регресійних моделей.")

        print(f"{'*' * 84}\nПоліноміальна регресія ")
        self.data_with_target = self.data_with_target_copy.copy()
        self.data_with_target = self.lof()
        X = self.data_with_target.drop("quality", axis=1)
        y = self.data_with_target['quality']
        x_poly = PolynomialFeatures(degree=2).fit_transform(X)
        print(self.basic_linear_regression(X=x_poly, y=y, print_results=False))
        print(f"Маємо незначне погіршення відносно лінійної регресії із застосуванням LOF для видалення викидів.\n"
              f"Визначимо значення коефіцієнту детермінацію при різних ступенях в моделі поліноміальної регресії.")
        print(f"{'*' * 84}\nЗалежність R2 від ступеню регресії")
        plot_data = [
            (n, self.basic_linear_regression(X=PolynomialFeatures(degree=n).fit_transform(X), y=y, print_results=False))
            for n in range(1, 5)
        ]
        sns.lineplot(data=pd.DataFrame(plot_data, columns=['degree', 'r2']), x='degree', y='r2')
        plt.title('Залежність R2 від ступеню регресії', fontsize=14)
        plt.show()
        print(f"{'*' * 84}\nВикористання моделі трансформації сплайнами. Функція сплайнів - поліноміальна")
        self.basic_linear_regression(X=SplineTransformer(degree=2, n_knots=4).fit_transform(X), y=y)
        print(f"Застосування моделей поліноміальної регресії дозволило незначно покращити результати прогнозування, \n"
              f"але коефіцієнт детермінації все ще має доволі низьке значення.")

        print(f"{'*' * 84}\nЛогістична регресія")
        print(
            f"{'*' * 84}\nНа відміну від звичайної регресії, у методі логістичної регресії не проводиться передбачення значення "
            f"числової змінної виходячи з вибірки вихідних значень. \n"
            f"Натомість значенням функції є ймовірність того, що дане вихідне значення належить до певного класу. \n"
            f"З урахуванням того, що цільова змінна має незначну кількість унікальних значень: 5, 7, 8, 9, 10, 11, \n"
            f"при застосуванні моделі очикується суттєве покращення результатів у порівнянні з наведеним вище застосуванням лінійної регресії.")
        self.basic_logistic_regression(X=X, y=y)
        print(
            f"Як можна бачити, отримані результати суттєво краще за отримані раніше з використанням лінійної та поліноміальної регресії.\n"
            f"Подальше покращення передбачень можливе з використанням дерев рішень, методів градієнтного спуску та їх реалізацій: XGBoost, LGBM та інших.")


if __name__ == '__main__':
    app = Kursova()
    app.load()
    app.save()
    app.investigation()
    app.modeling()
