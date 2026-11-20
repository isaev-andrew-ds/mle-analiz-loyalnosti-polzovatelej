import os
from dotenv import load_dotenv
from data_processor import DataProcessor

load_dotenv()

db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

query = '''
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
''' 

tenge_url = 'https://code.s3.yandex.net/datasets/final_tickets_tenge_df.csv'

data_processor = DataProcessor(
    df_config={
        'host': db_host,
        'port': db_port,
        'name': db_name,
        'user': db_user,
        'password': db_password,
    },
    query=query,
    tenge_url=tenge_url,
)

data_processor.connect()
df = data_processor.get_df()
tenge_df = data_processor.get_tenge_df()

# Конвертация курса валют в рубли
df = data_processor.convert_kzt_to_rub(df, tenge_df)

# Очистка выбросов в столбце revenue_rub
df = data_processor.clear_outliers_in_revenue_rub(df)

# Очистка полных дубликатов
df = data_processor.clear_complete_duplicates(df)

# Очистка отрицательных значений в столбце revenue
df = data_processor.clear_negative_revenue(df)

print(df.head())

# Создание датафрейма профилей пользователей
profiles_df = data_processor.create_users_profiles(df)

print(profiles_df.head())

# Показание статистики первых шагов профилей пользователей
data_processor.show_profiles_first_event_type_main_stat(profiles_df)
data_processor.show_profiles_first_device_type_stat(profiles_df)
data_processor.show_profiles_first_region_name_stat(profiles_df)

# Показание статистики профилей пользователей с двумя заказами и более по типу первого мероприятия
data_processor.show_profiles_with_two_orders_by_first_event_type_main(profiles_df)
# Показание статистики профилей пользователей с двумя заказами и более по типу первого устройства
data_processor.show_profiles_with_two_orders_by_first_device_type(profiles_df)
# Показание статистики профилей пользователей с двумя заказами и более по региону первого мероприятия
data_processor.show_profiles_with_two_orders_by_first_region_name(profiles_df)

# Проверка первой гипотезы
data_processor.show_first_hypothesis(profiles_df)
# Проверка второй гипотезы
data_processor.show_second_hypothesis(profiles_df)

# Показание разницы в средней выручке между профилями с одним заказом и с двумя и более
data_processor.show_diff_avg_revenue_between_profiles_types(profiles_df)

# Показание статистики профилей пользователей по дням недели первого заказа
data_processor.show_profiles_first_order_day_week_stat(profiles_df)

# Показание матрицы корреляции между признаками профилей пользователей
orders_count_correlation_matrix = data_processor.show_profiles_correlation_matrix(profiles_df)

print(orders_count_correlation_matrix)

data_processor.show_heat_map_correlation(orders_count_correlation_matrix, "Матрица корреляции между признаками профилей пользователей")