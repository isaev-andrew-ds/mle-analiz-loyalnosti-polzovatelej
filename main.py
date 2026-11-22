import os
from dotenv import load_dotenv
from src.processing import DataProcessor

load_dotenv()

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PWD")

query = """
    -- Настройка параметра synchronize_seqscans важна для проверки
    WITH set_config_precode AS (
    SELECT set_config('synchronize_seqscans', 'off', true)
    )
    -- Напишите ваш запрос ниже
    SELECT
    u.user_id,
    u.device_type_canonical,
    u.order_id,
    u.created_dt_msk AS order_dt,
    u.created_ts_msk AS order_ts,
    u.currency_code,
    u.revenue,
    u.tickets_count,
    (u.created_dt_msk::date - (LAG(u.created_dt_msk) OVER (PARTITION BY u.user_id ORDER BY created_dt_msk)::date)) AS days_since_prev,
    u.event_id,
    e.event_name_code AS event_name,
    e.event_type_main,
    u.service_name,
    r.region_name,
    c.city_name
    FROM afisha.purchases u
    LEFT JOIN afisha.events e USING(event_id)
    LEFT JOIN afisha.city c USING(city_id)
    LEFT JOIN afisha.regions r USING(region_id)
    WHERE e.event_type_main != 'фильм' AND (device_type_canonical = 'mobile' OR device_type_canonical = 'desktop')
    ORDER BY user_id 
    """

tenge_url = "https://code.s3.yandex.net/datasets/final_tickets_tenge_df.csv"


if __name__ == "__main__":
    data_processor = DataProcessor(
        {
            "host": db_host,
            "port": db_port,
            "db": db_name,
            "user": db_user,
            "pwd": db_password,
        },
        query,
        tenge_url,
    )

    data_processor.connect()

    print("Подключение к базе данных успешно")

    df = data_processor.get_df()
    tenge_df = data_processor.get_tenge_df()

    print("Данные получены: датасет заказов и датасет курсов валют")

    # Конвертация курса валют в рубли
    df = data_processor.convert_kzt_to_rub(df, tenge_df)

    # Сохраняем первоначальный размер датафрейма
    data_processor.save_df_size(df)

    # Очистка выбросов в столбце revenue_rub
    df = data_processor.clear_outliers_in_revenue_rub(df)

    # Очистка полных дубликатов
    df = data_processor.clear_complete_duplicates(df)

    # Очистка отрицательных значений в столбце revenue
    df = data_processor.clear_negative_revenue(df)

    # Итоги предобработки
    data_processor.log_deleted_rows_size(df)

    print(f"""

Итоги предобработки:
\nБыл добавлен новый столбец revenue_rub, в котором выручка в тенге была преобразована в рубли, пропусков в новом столбце при преобразований валют не оказалось;
\nТипы данных остались, все типы корректно обозначают данные;
\nПропуски - в столбце event_type_main значение другое обозначает пропуск, остальные столбцы не имеют пропусков;
\nБыли отфильтрованы данные по полю revenue_rub, остались строки, где значение revenue_rub было меньше 99 процентиля, убралось {data_processor.get_clear_stat("outliers_revenue_rub")['procent']}% строк, {data_processor.get_clear_stat("outliers_revenue_rub")['count']} штук;
\nБыли удалены записи дубликаты с разным уникальным идентификатором order_id в размере {data_processor.get_clear_stat("complete_duplicates")['count']} штук
\nБыли удалены записи с отрицательной выручкой в размере {data_processor.get_clear_stat("negative_revenue")['count']} штук
\nВсего было удалено {data_processor.get_deleted_rows_size(df)} записей, что составляет {data_processor.get_deleted_rows_procent(df)}% процентов от исходного датафрейма
    """)

    print(df.head())

    # Создание датафрейма профилей пользователей
    profiles_df = data_processor.create_users_profiles(df)

    print(profiles_df.head())

    # Очистка профилей от выбросов в количестве заказов
    profiles_df = data_processor.clear_outliers_in_count_orders(profiles_df)

    # Показание статистики первых шагов профилей пользователей
    top_first_event_types = data_processor.show_profiles_first_event_type_main_stat(
        profiles_df
    )
    top_devices = data_processor.show_profiles_first_device_type_stat(profiles_df)
    top_regions = data_processor.show_profiles_first_region_name_stat(profiles_df)

    # Показание статистики профилей пользователей с двумя заказами и более по типу первого мероприятия
    returned_event_types = (
        data_processor.show_profiles_with_two_orders_by_first_event_type_main(
            profiles_df
        )
    )
    # Показание статистики профилей пользователей с двумя заказами и более по типу первого устройства
    returned_device_type = (
        data_processor.show_profiles_with_two_orders_by_first_device_type(profiles_df)
    )
    # Показание статистики профилей пользователей с двумя заказами и более по региону первого мероприятия
    data_processor.show_profiles_with_two_orders_by_first_region_name(profiles_df)

    # Проверка первой гипотезы
    data_processor.show_first_hypothesis(profiles_df)
    # Проверка второй гипотезы
    data_processor.show_second_hypothesis(profiles_df)

    # Показание разницы в средней выручке между профилями с одним заказом и с двумя и более
    data_processor.show_diff_avg_revenue_between_profiles_types(profiles_df)

    # Показание статистики профилей пользователей по дням недели первого заказа
    first_day_users_stat = data_processor.show_day_of_week_stat(profiles_df)

    # Показание матрицы корреляции между признаками профилей пользователей
    orders_count_correlation_matrix = data_processor.matrix_correlation_by_orders_count(
        profiles_df
    )

    print(orders_count_correlation_matrix)

    data_processor.show_heat_map_correlation(
        orders_count_correlation_matrix,
        "Матрица корреляции между признаками профилей пользователей",
    )

    print(f"""
    В ходе исследования была проведена обработка данных: выручки приведены в рублевый эквивалент по актуальным курсам, удалены выбросы выручек по 99 процентилю, удалены явные дубликаты по всем столбцам кроме уникального order_id заказа, а также исключены записи из сводки профилей пользователей с аномальными значениями количества заказов по 99 процентилю.

    \nВсего профилей пользователей после обработки оказалось {profiles_df.shape[0]}. Пользователи по количеству заказов распределены крайне неравномерно - большинство пользователей имеют количество заказов от 0 до 10, при это максимальное количество заказов {profiles_df['count_orders'].max()}. При этом, первым мероприятием часто оказывается {", ".join(top_first_event_types.index[:3])}. В первых заказах такие доли устройств: {round(top_devices['mobile'] / top_devices.sum() * 100, 2)}% - мобильные устройства, {round(top_devices['desktop'] / top_devices.sum() * 100,2)}% - десктоп, а первым регионом часто выступает {top_regions.index[0]} {round(top_regions[top_regions.index[0]] / profiles_df["user_id"].nunique() * 100, 2)}% и {top_regions.index[1]} {round(top_regions[top_regions.index[1]] / profiles_df["user_id"].nunique() * 100, 2)}%.

    \nЧаще всего пользователи возвращаются при первом мероприятия типов: {", ".join(returned_event_types.index[:3])}; доля возвращаемых пользователей среди регинов не зависит от количества пользователей, и поэтому в регионах с низкой посещаемостью мероприятий доля возвращаемых пользователей может быть выше, чем с высокой посещаемостью. При {"мобильных" if returned_device_type.index[0] == 'mobile' else "десктоп"} устройствах на {round((returned_device_type[returned_device_type.index[0]] - returned_device_type[returned_device_type.index[1]]) * 100,2)}% чаще возвращаются именно в этом сегменте.

    \nЧаще всего временные характеристики не особо влияют на возвращение пользователей: можно лишь выделить, что чаще возвращаются к покупками пользователи, у которых день недели первого заказа были {", ".join(first_day_users_stat.index[:3])}.

    \nБыло выявлено при помощи расчёта корреляции phi_k, что первый день заказа, последний день заказа, возвращение пользователя и среднее количество дней между заказами наиболее влиют на число заказов.В первую очередь следует обратить внимание на пользователей, которые делают заказ с типом мероприятия {", ".join(returned_event_types.index[:3])} - во всех них высокая доля возвращающихся пользователей.

    \nМаркетинговые усилия стоит обратить на пользователей, которые делают заказы на мероприятия в регионы {", ".join(top_first_event_types.index[:3])}, поскольку они часто являются точкой входа.

    \nТакже стоит сделать усилия на пользователей, делающих первый заказ с {"мобильных" if returned_device_type.index[0] == 'mobile' else "десктоп"} устройств: они составляют высокую долю точки входа, но при этом доля возвращающихся пользователей на {round((returned_device_type[returned_device_type.index[0]] - returned_device_type[returned_device_type.index[1]]) * 100,2)}% выше

    \nСтоит раскрыть мероприятия типов другие, так как данные мероприятия также являются основной точкой входа, но при этом доля возвращающихся пользователей невелика

    \nДля продолжения исследования следует взять более большой период, а также, как говорилось выше, раскрыть тип мероприятий другие
    """)
