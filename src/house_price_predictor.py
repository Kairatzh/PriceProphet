import os
import logging
import pickle
import pandas as pd
from typing import Dict, List
from dotenv import load_dotenv
import telebot
from telebot.types import Message
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="../house_price_predictor.log",
    filemode="a"
)
class Config:
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    RF_PARAMS = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5, 10]
    }
    GBR_PARAMS = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 4, 5]
    }
    DATA_PATH = "../data/house_price_regression_dataset.csv"
    MODEL_DIR = "../models"
    MODEL_PATH = os.path.join(MODEL_DIR, "house_price_best_model.pkl")
    BOT_VERSION = "1.0.0"
FIELDS = [
    'Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms',
    'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality'
]
FIELD_DESCRIPTIONS = {
    'Square_Footage': 'Площадь дома (кв. футы)',
    'Num_Bedrooms': 'Количество спален',
    'Num_Bathrooms': 'Количество ванных комнат',
    'Year_Built': 'Год постройки',
    'Lot_Size': 'Размер участка (кв. футы)',
    'Garage_Size': 'Размер гаража (кол-во машин)',
    'Neighborhood_Quality': 'Качество района (1-10)'
}
def train_model():
    """Обучение и сохранение модели"""
    try:
        data_house = pd.read_csv(Config.DATA_PATH)
        logging.info("Данные успешно загружены")
    except FileNotFoundError:
        logging.error(f"Файл данных не найден: {Config.DATA_PATH}")
        print(f"❌ Ошибка: Файл {Config.DATA_PATH} не найден")
        exit(1)
    data_house.drop_duplicates(inplace=True)
    logging.info("Дубликаты удалены")
    price_column = next((col for col in data_house.columns if 'price' in col.lower()), None)

    if not price_column:
        logging.error("Не найдена колонка с ценой дома")
        print("❌ Ошибка: Не найдена колонка с ценой дома")
        exit(1)
    logging.info(f"Найдена колонка с ценой: {price_column}")

    X = data_house.drop(columns=[price_column])
    y = data_house[price_column]
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_features)
    ])
    def create_pipeline(model):
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

    def tune_model(model_class, params, X, y):
        pipeline = create_pipeline(model_class(random_state=Config.RANDOM_STATE))
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=10,
            cv=Config.CV_FOLDS,
            scoring='r2',
            n_jobs=-1,
            verbose=1,
            random_state=Config.RANDOM_STATE
        )
        search.fit(X, y)
        return search.best_estimator_

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE
    )

    logging.info("Начало обучения моделей!!")
    rf_model = tune_model(RandomForestRegressor,
                          Config.RF_PARAMS,
                          X_train, y_train)
    gbr_model = tune_model(GradientBoostingRegressor,
                           Config.GBR_PARAMS,
                           X_train, y_train)

    rf_score = r2_score(y_test,
                        rf_model.predict(X_test))
    gbr_score = r2_score(y_test,
                         gbr_model.predict(X_test))

    best_model = rf_model if rf_score > gbr_score else gbr_model

    best_score = max(rf_score, gbr_score)

    logging.info(f"Лучшая модель: {'RandomForest' if rf_score > gbr_score else 'GradientBoosting'}, R2: {best_score}")

    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    try:
        with open(Config.MODEL_PATH, "wb") as f:
            pickle.dump(best_model, f)
        logging.info(f"Модель успешно сохранена в {Config.MODEL_PATH}")
        print(f"✅ Модель обучена и сохранена в {Config.MODEL_PATH}")
    except Exception as e:
        logging.error(f"Ошибка сохранения модели: {e}")
        print(f"❌ Ошибка сохранения модели: {e}")
        exit(1)

    return best_model

class HousePricePredictor:
    def __init__(self):
        self.bot = None
        self.model = None
        self._initialize()

    def _initialize(self):
        try:
            load_dotenv()
            bot_token = os.getenv('BOT_TOKEN')
            if not bot_token:
                raise ValueError("Токен бота не найден в .env!")
            self.bot = telebot.TeleBot(bot_token)
            logging.info(f"Бот инициализирован с токеном: {bot_token[:10]}...")

            if os.path.exists(Config.MODEL_PATH) and os.path.getsize(Config.MODEL_PATH) > 0:
                try:
                    with open(Config.MODEL_PATH, "rb") as f:
                        self.model = pickle.load(f)
                    logging.info("Модель успешно загружена из файла")
                except Exception as e:
                    logging.warning(f"Ошибка загрузки модели: {e}. Переобучаем модель...")
                    print(f"Модель повреждена, запускаем обучение...")
                    self.model = train_model()
            else:
                print("Модель не найдена или пуста, запускаем обучение...")
                self.model = train_model()

            self._register_handlers()
            logging.info("Инициализация завершена")
        except Exception as e:
            logging.error(f"Ошибка инициализации: {e}")
            print(f"❌ Ошибка инициализации: {e}")
            raise

    def _register_handlers(self):
        @self.bot.message_handler(commands=['start'])
        def handle_start(message: Message):
            self._send_welcome(message)

        @self.bot.message_handler(commands=['help'])
        def handle_help(message: Message):
            self._send_help(message)

        @self.bot.message_handler(commands=['fields'])
        def handle_fields(message: Message):
            self._send_fields(message)

        @self.bot.message_handler(commands=['predict'])
        def handle_predict(message: Message):
            self._start_prediction(message)

        @self.bot.message_handler(commands=['about'])
        def handle_about(message: Message):
            self._send_about(message)

        @self.bot.message_handler(func=lambda m: True)
        def handle_message(message: Message):
            self._process_prediction(message)

    def _send_welcome(self, message: Message):
        welcome_text = (
            "👋 Добро пожаловать! Я бот для предсказания цен на недвижимость.\n\n"
            "🏠 Помогу оценить стоимость дома.\n"
            "📊 Используйте /predict для предсказания.\n"
            "❓ Нужна помощь? Введите /help.\n"
            "ℹ️ Информация о боте: /about"
        )
        self.bot.reply_to(message, welcome_text)

    def _send_help(self, message: Message):
        help_text = (
            "🔍 Команды:\n"
            "/start - Начать\n"
            "/predict - Предсказать цену\n"
            "/fields - Описание полей\n"
            "/help - Показать справку\n"
            "/about - О боте"
        )
        self.bot.reply_to(message, help_text)

    def _send_fields(self, message: Message):
        text = "📋 Параметры для предсказания:\n\n"
        for i, field in enumerate(FIELDS, 1):
            text += f"{i}. {FIELD_DESCRIPTIONS[field]}\n"
        text += "\nПример: 2500, 3, 2, 2010, 5000, 2, 8"
        self.bot.reply_to(message, text)

    def _send_about(self, message: Message):
        about_text = (
            "ℹ️ **О боте**\n\n"
            "🏠 Предсказывает стоимость дома.\n"
            f"📌 Версия: {Config.BOT_VERSION}\n"
            "👨‍💻 Автор: [Ваше имя]\n"
            "📅 Создан: Март 2025"
        )
        self.bot.reply_to(message, about_text)
        logging.info(f"Пользователь {message.from_user.id} запросил /about")

    def _start_prediction(self, message: Message):
        text = "📝 Введите характеристики через запятую:\n\n"
        for i, field in enumerate(FIELDS, 1):
            text += f"{i}. {FIELD_DESCRIPTIONS[field]}\n"
        text += "\nПример: 2500, 3, 2, 2010, 5000, 2, 8"
        self.bot.reply_to(message, text)

    def _validate_input(self, values: List[str]) -> Dict:
        if len(values) != len(FIELDS):
            raise ValueError(f"Ожидается {len(FIELDS)} параметров, получено {len(values)}")
        try:
            validated = {
                'Square_Footage': float(values[0]),
                'Num_Bedrooms': int(values[1]),
                'Num_Bathrooms': int(values[2]),
                'Year_Built': int(values[3]),
                'Lot_Size': float(values[4]),
                'Garage_Size': int(values[5]),
                'Neighborhood_Quality': int(values[6])
            }
            if not 1 <= validated['Neighborhood_Quality'] <= 10:
                raise ValueError("Качество района: 1-10")
            if validated['Year_Built'] > 2025 or validated['Year_Built'] < 1900:
                raise ValueError("Год постройки: 1900-2025")
            if any(v < 0 for v in validated.values()):
                raise ValueError("Значения не могут быть отрицательными")
            return validated
        except ValueError as e:
            raise ValueError(f"Ошибка ввода: {str(e)}")

    def _process_prediction(self, message: Message):
        try:
            values = [v.strip() for v in message.text.split(',')]
            validated_data = self._validate_input(values)
            input_df = pd.DataFrame([validated_data])
            prediction = self.model.predict(input_df)[0]

            response = (
                "🏠 Результат:\n\n"
                f"💰 Цена: ${prediction:,.2f}\n\n"
                "📋 Характеристики:\n"
            )
            for field, value in zip(FIELDS, values):
                response += f"➡️ {FIELD_DESCRIPTIONS[field]}: {value}\n"
            self.bot.reply_to(message, response)
            logging.info(f"Предсказание для {message.from_user.id}: ${prediction:,.2f}")
        except ValueError as e:
            self.bot.reply_to(message, f"❌ Ошибка: {str(e)}\n\nИспользуйте /predict")
        except Exception as e:
            self.bot.reply_to(message, "❌ Ошибка обработки. Попробуйте снова.")
            logging.error(f"Ошибка предсказания: {e}")

    def run(self):
        logging.info("Запуск бота...")
        print("🤖 Бот запущен...")
        self.bot.polling(none_stop=True, interval=0, timeout=20)

def main():
    try:
        predictor = HousePricePredictor()
        predictor.run()
    except Exception as e:
        logging.critical(f"Критическая ошибка: {e}")
        print(f"❌ Критическая ошибка: {e}")
        exit(1)

if __name__ == "__main__":
    main()