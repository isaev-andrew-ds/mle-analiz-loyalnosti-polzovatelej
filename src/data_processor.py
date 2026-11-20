from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy import Engine
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from phik import phik_matrix


# Класс для предобработки данных для анализа
class DataProcessor:

    def __init__(
        self,
        df_config: dict,
        query: str,
        tenge_url: str,
    ):
        self.df_config = df_config
        self.query = query
        self.tenge_url = tenge_url

    # Метод для подключения к базе данных
    def connect(self) -> Engine:
        connection_string = "postgresql://{}:{}@{}:{}/{}".format(
            self.df_config["user"],
            self.df_config["pwd"],
            self.df_config["host"],
            self.df_config["port"],
            self.df_config["db"],
        )
        self.engine = create_engine(connection_string)

    # Метод для получения датафрейма из базы данных
    def get_df(self) -> pd.DataFrame:
        # Возвращаем датафрейм, а не сохраняем в атрибут класса
        return pd.read_sql_query(self.query, con=self.engine)

    # Метод для получения датафрейма с курсами валют
    def get_tenge_df(self) -> pd.DataFrame:
        # Возвращаем датафрейм, а не сохраняем в атрибут класса
        return pd.read_csv(self.tenge_url)

    # Метод предобработки для конвертации курса валют в рубли
    def convert_kzt_to_rub(
        self, df: pd.DataFrame, tenge_df: pd.DataFrame
    ) -> pd.DataFrame:
        # Копируем датафреймы, чтобы не было побочных эффектов из-за ссылок
        df = df.copy()
        tenge_df = tenge_df.copy()

        tenge_df["data"] = pd.to_datetime(tenge_df["data"], format="%Y-%m-%d")

        df_with_tenge_curs = df.merge(
            tenge_df, how="left", left_on="order_dt", right_on="data"
        )

        df["revenue_rub"] = df_with_tenge_curs.apply(self.__rub_to_tenge, axis=1)

        # Выводим количество строк без пропусков и пропусков нового столбца
        print(
            f"Количество строк без пропусков нового столбца: {df['revenue_rub'].count()}"
        )
        print(f"Количество пропусков нового столбца: {df['revenue_rub'].isna().sum()}")
        return df

    # Метод предобработки для очистки выбросов в столбце revenue_rub
    def clear_outliers_in_revenue_rub(self, df: pd.DataFrame) -> pd.DataFrame:
        # Копируем датафрейм, чтобы не было побочных эффектов из-за ссылок
        df = df.copy()

        count_rows_before_filters = df.shape[0]

        df = df[df["revenue_rub"].quantile(0.99) > df["revenue_rub"]]

        self.__log_clear_info(
            count_rows_before_filters, df.shape[0], "выбросов в столбце revenue_rub"
        )

        return df

    # Метод предобработки для очистки полных дубликатов
    def clear_complete_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        # Копируем датафрейм, чтобы не было побочных эффектов из-за ссылок
        df = df.copy()

        count_rows_before_filters = df.shape[0]

        df = df.drop_duplicates(
            subset=[
                "user_id",
                "device_type_canonical",
                "order_dt",
                "order_ts",
                "currency_code",
                "revenue",
                "tickets_count",
                "days_since_prev",
                "event_id",
                "event_name",
                "event_type_main",
                "service_name",
                "region_name",
                "city_name",
                "revenue_rub",
            ]
        )

        self.__log_clear_info(
            count_rows_before_filters, df.shape[0], "полных дубликатов, кроме order_id"
        )

        return df

    # Метод предобработки для очистки отрицательных значений в столбце revenue
    def clear_negative_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        # Копируем датафрейм, чтобы не было побочных эффектов из-за ссылок
        df = df.copy()

        count_rows_before_filters = df.shape[0]

        df = df[df["revenue"] >= 0]

        self.__log_clear_info(
            count_rows_before_filters,
            df.shape[0],
            "отрицательных значений в столбце revenue",
        )

        return df

    # Метод создания датафрейма профилей пользователей
    def create_users_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        # Копируем датафрейм, чтобы не было побочных эффектов из-за ссылок
        df = df.copy()

        df = df.sort_values(by=["order_ts", "revenue_rub", "tickets_count"])

        profiles_df = (
            df.groupby("user_id")
            .agg(
                {
                    "order_dt": ["min", "max"],
                    "order_ts": ["min", "max"],
                    "order_id": "count",
                    "revenue_rub": "mean",
                    "days_since_prev": "mean",
                }
            )
            .reset_index()
        )

        profiles_df.columns = [
            "user_id",
            "first_order_dt",
            "last_order_dt",
            "first_order_ts",
            "last_order_ts",
            "count_orders",
            "mean_revenue",
            "mean_days_between_orders",
        ]

        profiles_df = profiles_df.sort_values(by="first_order_dt")

        profiles_df = profiles_df.merge(
            df,
            left_on=["user_id", "first_order_ts"],
            right_on=["user_id", "order_ts"],
            how="left",
        )

        profiles_df = profiles_df.rename(
            columns={
                "device_type_canonical": "first_order_device_type",
                "region_name": "first_order_region_name",
                "event_type_main": "first_event_type_main",
            }
        )

        profiles_df = profiles_df[
            [
                "user_id",
                "first_order_device_type",
                "first_order_region_name",
                "first_event_type_main",
                "first_order_dt",
                "last_order_dt",
                "first_order_ts",
                "last_order_ts",
                "count_orders",
                "mean_revenue",
                "mean_days_between_orders",
            ]
        ]

        print(
            f"Количество дубликатов user_id в датафрейме профилей: {profiles_df.duplicated(subset='user_id').sum()}"
        )

        # Добавление бинарного признака is_two
        profiles_df["is_two"] = np.where(profiles_df["count_orders"] >= 2, 1, 0)

        return profiles_df

    # Метод предобработки для очистки выбросов в столбце count_orders профилей пользователей
    def clear_outliers_in_count_orders(self, profiles_df: pd.DataFrame) -> pd.DataFrame:
        # Копируем датафрейм, чтобы не было побочных эффектов из-за ссылок
        profiles_df = profiles_df.copy()

        old_count_rows = profiles_df.shape[0]

        # Очистка данных по 99 процентилю количества заказов

        profiles_df = profiles_df[
            profiles_df["count_orders"] < profiles_df["count_orders"].quantile(0.99)
        ]

        self.__log_clear_info(
            old_count_rows,
            profiles_df.shape[0],
            "выбросов в столбце count_orders профилей пользователей",
        )

        return profiles_df

    # Метод для показа статистики первых шагов профилей пользователей по типу первого мероприятия
    def show_profiles_first_event_type_main_stat(
        self, profiles_df: pd.DataFrame
    ) -> None:
        self.__first_steps_profiles_stat(
            profiles_df,
            "first_event_type_main",
            "Количество пользователей в первом мероприятии",
            "Тип первого мероприятия",
            "Количество пользователей",
        )

        plt.show()

    # Метод для показа статистики первых шагов профилей пользователей по типу первого устройства
    def show_profiles_first_device_type_stat(self, profiles_df: pd.DataFrame) -> None:
        self.__first_steps_profiles_stat(
            profiles_df,
            "first_device_type",
            "Количество пользователей в первом устройстве",
            "Устройство первого мероприятия",
            "Количество пользователей",
        )
        plt.show()

    # Метод для показа статистики первых шагов профилей пользователей по региону первого мероприятия
    def show_profiles_first_region_name_stat(self, profiles_df: pd.DataFrame) -> None:
        self.__first_steps_profiles_stat(
            profiles_df,
            "first_region_name",
            "Количество пользователей в первом регионе",
            "Регион первого мероприятия",
            "Количество пользователей",
        )
        plt.show()

    # Метод для показа статистики профилей пользователей с двумя заказами и более по типу первого мероприятия
    def show_profiles_with_two_orders_by_first_event_type_main(
        self, profiles_df: pd.DataFrame
    ) -> None:
        self.__profiles_stat_per_segment_with_two_orders(
            profiles_df,
            "first_event_type_main",
        )
        plt.show()

    # Метод для показа статистики профилей пользователей с двумя заказами и более по типу первого устройства
    def show_profiles_with_two_orders_by_first_device_type(
        self, profiles_df: pd.DataFrame
    ) -> None:
        self.__profiles_stat_per_segment_with_two_orders(
            profiles_df, "first_device_type", color="lime"
        )
        plt.show()

    # Метод для показа статистики профилей пользователей с двумя заказами и более по региону первого мероприятия
    def show_profiles_with_two_orders_by_first_region_name(
        self, profiles_df: pd.DataFrame
    ) -> None:
        self.__profiles_stat_per_segment_with_two_orders(
            profiles_df, "first_region_name", color="pink"
        )
        plt.show()

    # Метод для проверки первой гипотезы
    def show_first_hypothesis(self, profiles_df: pd.DataFrame) -> None:
        first_sports_concerts_users_types = profiles_df[
            (profiles_df["first_event_type_main"] == "спорт")
            | (profiles_df["first_event_type_main"] == "концерты")
        ]

        first_sports_concerts_users_types = first_sports_concerts_users_types.groupby(
            "first_event_type_main"
        )["is_two"].mean()

        print(first_sports_concerts_users_types)

        first_sports_concerts_users_types.plot(
            kind="bar",
            title="Доля пользователей, которые сделали повторный заказ и у которых первое мероприятие - спорт или концерт",
            xlabel="Тип мероприятия",
            ylabel="Доля пользователей",
            figsize=(20, 10),
            rot=0,
            color="purple",
            edgecolor="black",
        )

        plt.show()

    # Метод для проверки второй гипотезы
    def show_second_hypothesis(self, profiles_df: pd.DataFrame) -> None:
        top_regions_stat = (
            profiles_df.groupby("first_order_region_name")
            .agg({"user_id": "nunique", "is_two": "mean"})
            .sort_values(by="user_id", ascending=False)
        )

        top_regions_stat["is_two"] = round(top_regions_stat["is_two"] * 100, 2)

        top_regions_stat = top_regions_stat.rename(
            columns={
                "user_id": "Количество пользователей",
                "is_two": "Доля вернувшихся пользователи",
            }
        )

        print("=== Доля вернувшихся пользователей ===")
        print(top_regions_stat)

        pd.concat([top_regions_stat[:10], top_regions_stat[-10:]]).plot(
            kind="barh",
            title="Процент вернувшихся пользователей в топ регионах по сравнению с количеством пользователей",
            xlabel="Регион",
            ylabel="Процент пользователей",
            figsize=(20, 14),
            edgecolor="black",
        )

        plt.show()

    # Метод для показа разницы в средней выручке между профилями с одним заказом и с двумя и более
    def show_diff_avg_revenue_between_profiles_types(
        self, profiles_df: pd.DataFrame
    ) -> None:
        plt.figure(figsize=(20, 14))

        sns.histplot(
            data=profiles_df,
            x="mean_revenue",
            hue="is_two",
            alpha=0.67,
            bins=30,
            kde=True,
            stat="density",
        )

        plt.title(
            "Распределение средней выручки в рублях среди клиентов с одним заказом, и с двумя и более"
        )
        plt.xlabel("Выручка в рублях")
        plt.ylabel("Плотность")

        plt.show()

    def show_day_of_week_stat(self, profiles_df: pd.DataFrame) -> None:
        first_day_df = profiles_df.copy()

        first_day_df["first_order_day_week"] = first_day_df[
            "first_order_dt"
        ].dt.day_of_week.apply(self.__day_of_week_to_str)

        first_day_users_stat = (
            first_day_df.groupby("first_order_day_week")
            .agg({"user_id": "count", "is_two": "mean"})
            .sort_values(by="is_two", ascending=False)
            .rename(
                columns={
                    "user_id": "Количество пользователей",
                    "is_two": "Доля пользователей, совершивших повторные заказы",
                }
            )
        )

        print(
            "=== Количество пользователей, совершивших первый заказ в определённый день недели, и доля пользователей, совершивших повторные заказы ==="
        )
        print(first_day_users_stat)

        first_day_users_stat.plot(
            kind="bar",
            title="Количество пользователей, совершивших первый заказ в определённый день недели, и доля пользователей, совершивших повторные заказы",
            xlabel="День недели первого заказа",
            ylabel=[
                "Количество пользователей",
                "доля пользователей с повторными заказами",
            ],
            subplots=True,
            figsize=(20, 16),
            rot=True,
        )

        plt.show()

    # Метод для показа матрицы корреляции между признаками профилей пользователей с количеством заказов
    def matrix_correlation_by_orders_count(
        self, profiles_df: pd.DataFrame
    ) -> pd.DataFrame:
        user_profile_corr = (
            profiles_df[
                [
                    "first_order_device_type",
                    "first_order_region_name",
                    "first_event_type_main",
                    "first_order_dt",
                    "last_order_dt",
                    "count_orders",
                    "mean_revenue",
                    "mean_days_between_orders",
                    "is_two",
                ]
            ]
            .phik_matrix(
                interval_cols=[
                    "count_orders",
                    "mean_revenue",
                    "mean_days_between_orders",
                ]
            )["count_orders"]
            .sort_values(ascending=False)
        )

        user_profile_corr = (
            user_profile_corr[1:]
            .reset_index()
            .rename(
                columns={
                    "index": "Признак профиля",
                    "count_orders": "Количество заказов пользователя",
                },
            )
            .set_index("Признак профиля")
            .rename(
                index={
                    "first_order_region_name": "Регион первого заказа",
                    "first_order_dt": "Дата первого заказа",
                    "last_order_dt": "Дата последнего заказа",
                    "mean_revenue": "Средняя выручка заказов пользователя",
                    "first_event_type_main": "Тип первого мероприятия",
                    "first_order_device_type": "Тип устройства первого заказа",
                    "mean_days_between_orders": "Среднее количество дней между заказами пользователя",
                    "is_two": "Флаг возвращения",
                }
            )
        )
        return user_profile_corr

    # Метод для показа тепловой карты корреляции между признаками профилей пользователей с количеством заказов
    def show_heat_map_correlation(
        self, data: pd.DataFrame, title: str
    ) -> None:
        sns.heatmap(data=data, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1)

        plt.title(title)

        plt.show()

    """Приватные методы"""

    # Внутренний метод для логирования информации о очистке данных
    def __log_clear_info(
        self,
        count_rows_before_filters: int,
        count_rows_after_filters: int,
        operation="",
    ) -> None:
        print(
            (f"=== Очистка {operation} ===") if operation else "=== Очистка данных ==="
        )
        print(f"Количество строк до очистки: {count_rows_before_filters}")
        print(f"Количество строк после очистки: {count_rows_after_filters}")
        print(f"Удалено строк: {count_rows_before_filters - count_rows_after_filters}")
        print(
            f"Процент удалённых строк: {round((count_rows_before_filters - count_rows_after_filters)/count_rows_before_filters * 100, 2)}%"
        )

    # Внутренний метод для конвертации рублей в тенге
    def __rub_to_tenge(self, row: pd.Series) -> float:
        currency_code = row["currency_code"]
        if currency_code == "kzt":
            return row["revenue"] * row["curs"] / row["nominal"]
        return row["revenue"]

    # Внутренний метод для статистики первых шагов профилей пользователей
    def __first_steps_profiles_stat(
        self,
        profiles_df: pd.DataFrame,
        column: str,
        title: str,
        xlabel: str,
        ylabel: str,
        horizont: bool = False,
    ) -> None:
        unique_users_per_first_step = (
            profiles_df.groupby(column)["user_id"]
            .nunique()
            .sort_values(ascending=False)
        )

        print("=== Количество уникальных пользователей в каждой категории ===")
        print(unique_users_per_first_step)
        print()
        print("=== Доли от общего числа пользователей в процентах ===")
        print(
            round(
                unique_users_per_first_step / profiles_df["user_id"].nunique() * 100, 2
            )
        )

        unique_users_per_first_step = (
            unique_users_per_first_step[:15]
            if len(unique_users_per_first_step) > 15
            else unique_users_per_first_step[:]
        )

        unique_users_per_first_step.plot(
            kind=("bar" if not horizont else "barh"),
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=(20, 12),
            rot=0,
        )

    def __profiles_stat_per_segment_with_two_orders(
        self,
        profiles_df: pd.DataFrame,
        column: str,
        max_groups: int = 12,
        color: str = "aqua",
    ) -> None:
        grouped_segment = (
            profiles_df.groupby(column)["is_two"].mean().sort_values(ascending=False)
        )

        print(
            f"=== Доля пользователей, совершивших два и более заказа, в сегменте: {column}. В процентах ==="
        )
        print(grouped_segment)

        grouped_segment = (
            grouped_segment[:max_groups]
            if len(grouped_segment) > max_groups
            else grouped_segment[:]
        )

        grouped_segment.plot(
            kind="bar",
            title=f"Доля пользователей, совершивших два и более заказа по сегментам {column}",
            xlabel=f"Сегменты {column}",
            ylabel="Доля пользователей",
            figsize=(20, 12),
            rot=0,
            color=color,
            edgecolor="black",
        )

    # Внутренний метод для конвертации дня недели в строку
    def __day_of_week_to_str(self, day: int) -> str:
        if day == 0:
            return "пн"
        elif day == 1:
            return "вт"
        elif day == 2:
            return "ср"
        elif day == 3:
            return "чт"
        elif day == 4:
            return "пт"
        elif day == 5:
            return "сб"
        else:
            return "вс"
