import re
import time
import csv

import vk_api
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from itertools import zip_longest
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import re
import csv

def convert_views(views_str):
    match = re.search(r'(\d+[\.,]?\d*)(K)?\s*(просмотр|прочтений)', views_str)
    if match:
        number_str, is_thousand, _ = match.groups()
        number = float(number_str.replace(',', '.'))  # Преобразование запятой в точку для float
        if is_thousand:
            return int(number * 1000)
        else:
            return int(number)
    return 0
def scrape_data(driver, url, css_selector, text_selector, views_selector, max_scroll_count):
    driver.get(url)
    posts_data = []
    scroll_count = 0

    while scroll_count < max_scroll_count:
        post_divs = driver.find_elements(By.CSS_SELECTOR, css_selector)
        for post_div in post_divs:
            post_data = {}
            text_elements = post_div.find_elements(By.CSS_SELECTOR, text_selector)
            views_elements = post_div.find_elements(By.CSS_SELECTOR, views_selector)
            if text_elements and views_elements:
                post_data["text"] = text_elements[0].text
                post_data["views"] = convert_views(views_elements[0].text)
                post_data["likes"]=0
                posts_data.append(post_data)
            else:
                continue

        body = driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.END)
        time.sleep(2)  # небольшая задержка для загрузки контента

        scroll_count += 1

    return posts_data


def dzen(url_base, max_scroll_count=5):
    driver = webdriver.Chrome()
    try:
        # Скрейпинг видео
        video_url = url_base + "?tab=longs"
        video_data = scrape_data(
            driver,
            video_url,
            "div.feed__row._should-hide-margin._items-count_1",
            ".card-video-punch__title-25",
            ".zen-ui-common-layer-meta.card-video-punch__withLeftMargin-2h.card-video-punch__metaContainer-1A",
            max_scroll_count
        )

        # Скрейпинг статей
        article_url = url_base + "?tab=articles"
        article_data = scrape_data(
            driver,
            article_url,
            "div.feed__row._should-hide-margin._items-count_1._is-full-height",
            ".zen-ui-rich-text._theme_white",
            ".zen-ui-common-layer-meta.floor-channel-info__metaContainer-1N",    max_scroll_count)

        driver.quit()

        combined_data = video_data + article_data


        keys = combined_data[0].keys()
        with open('result.csv', 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(combined_data)
            output_file.close()
    except:
        print('error')


def vk(url,scroll_pause_time=1,max_scroll_count = 15,max_count=150):
    # Создаем сессию VK API
    vk_session = vk_api.VkApi(token='vk1.a.BixKz45O4nwKRSa16gay-w5m71IjKZVH2z9IV6AOe9X4t2435KSvMYV9xhl1_MhcHO47nl4NZ81nzEo9DGcYToU010aAXw9onuWUfEtNKe-T6M2s5jxz_ZzBZJIQXjDCCQunO6m-YVCUSmEsVP9RB9kTZj4-g6ZQfbj-sYnyBVrl4FCCtdw9z_3Jcx3Y1tMRk_8bQHR7YhkucA1ACf5tww')

    vk = vk_session.get_api()

    # Выделяем идентификатор пользователя или сообщества из URL

    with open('result.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'likes', 'views']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        try:
            group_short_name = url.split('/')[-1]
            group_info = vk.groups.getById(group_id=group_short_name)
            group_id = -group_info[0]['id']
            # Запрашиваем посты со стены
            response = vk.wall.get(owner_id=group_id, count=max_count)
            for item in response['items']:
                post_text = item['text']
                likes = item['likes']['count']
                views = item['views']['count'] if 'views' in item else 'N/A'

                writer.writerow({'text': post_text, 'likes': likes, 'views': views})

        except Exception as e:
            print(e)


while True:
    print('Вставьте ссылку :')
    inp_url=input()
    d=inp_url.split(sep='/')
    if d[2]=='vk.com':
        vk(inp_url)
    elif d[2]=='dzen.ru':
        dzen(inp_url,max_scroll_count=6)
    else:
        print('Error ans')



    df = pd.read_csv('result.csv')
    # Очистка данных
    df['likes'] = pd.to_numeric(df['likes'].replace('K', 'e3').replace('N/A', '0'), errors='coerce').fillna(0).astype(int)
    df['views'] = pd.to_numeric(df['views'].replace('K', 'e3').replace('N/A', '0'), errors='coerce').fillna(0).astype(int)
    russian_stop_words =[
        'а', 'без', 'более', 'больше', 'будет', 'будто', 'бы', 'был', 'была', 'были', 'было', 'быть',
        'в', 'вам', 'вами', 'вас', 'весь', 'вдоль', 'вдруг', 'везде', 'ведь', 'весь', 'вниз', 'внизу',
        'во', 'вокруг', 'вон', 'вот', 'впрочем', 'все', 'всегда', 'всего', 'всех', 'всю', 'вся', 'вы',
        'г', 'где', 'да', 'давай', 'давать', 'даже', 'для', 'до', 'достаточно', 'другая', 'другие',
        'других', 'друго', 'другое', 'другой', 'его', 'едим', 'едят', 'ее', 'если', 'есть', 'еще', 'ещё',
        'её', 'ж', 'же', 'жизнь', 'за', 'зачем', 'здесь', 'и', 'из', 'или', 'им', 'имеет', 'ими', 'имя',
        'иногда', 'их', 'к', 'каждая', 'каждое', 'каждые', 'каждый', 'как', 'какая', 'какой', 'кем',
        'когда', 'кого', 'которая', 'которого', 'которое', 'которой', 'которые', 'который', 'которых',
        'кто', 'куда', 'ли', 'лишь', 'лучше', 'люди', 'м', 'между', 'меня', 'мне', 'много', 'может',
        'можно', 'мой', 'моя', 'мы', 'на', 'над', 'надо', 'наконец', 'нас', 'не', 'него', 'нее', 'ней',
        'нельзя', 'нет', 'ни', 'нибудь', 'ниже', 'ним', 'них', 'ничего', 'но', 'новый', 'ноги', 'ночь',
        'ну', 'нужно', 'нх', 'о', 'об', 'оба', 'обычно', 'один', 'однако', 'одной', 'около', 'он',
        'она', 'они', 'оно', 'опять', 'особенно', 'от', 'ответить', 'откуда', 'отовсюду', 'отсюда', 'очень',
        'первый', 'перед', 'по', 'под', 'пожалуйста', 'позже', 'пока', 'понимать', 'понятно', 'пор',
        'пора', 'после', 'посреди', 'потом', 'почти', 'при', 'про', 'раз', 'разве', 'разве', 'с',
        'сам', 'сама', 'сами', 'самим', 'самих', 'само', 'самого', 'самой', 'самом', 'самому', 'саму',
        'самый', 'свое', 'своего', 'своей', 'свои', 'своим', 'своих', 'свою', 'себе', 'себя', 'сегодня',
        'сейчас', 'сказал', 'сказала', 'сказать', 'сказать', 'сказать', 'сможет', 'снова', 'со', 'собой',
        'собою', 'совсем', 'спасибо', 'сразу', 'среди', 'суть', 'считать', 'т', 'та', 'так', 'такая', 'также',
        'такие', 'такое', 'такой', 'там', 'твой', 'те', 'тебе', 'тебя', 'тем', 'теми', 'теперь', 'то',
        'тобой', 'тобою', 'товарищ', 'тогда', 'того', 'тоже', 'только', 'том', 'тот', 'точно', 'три',
        'ту', 'ты', 'у', 'уж', 'уже', 'хорошо', 'хотеть', 'хоть', 'хочешь', 'час', 'часто', 'часть', 'чего',
        'человек', 'чем', 'чему', 'через', 'четвертый', 'что', 'чтобы', 'чуть', 'шестой', 'шесть', 'эта',
        'эти', 'этим', 'этими', 'этих', 'это', 'этого', 'этой', 'этом', 'этому', 'этот', 'эту', 'я'
    ]

    # Анализ
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=russian_stop_words)
    tfidf = vectorizer.fit_transform(df['text'])
    nmf = NMF(n_components=15, random_state=1).fit(tfidf)
    feature_names = np.array(vectorizer.get_feature_names_out())
    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        topic_features = ', '.join(feature_names[topic.argsort()[:-10 - 1:-1]])
        print(f"Тема #{topic_idx + 1}: {topic_features}")
        topics.append(f"Тема #{topic_idx + 1}: {topic_features}")


    topic_results = nmf.transform(tfidf)
    df['topic'] = topic_results.argmax(axis=1) + 1


    plt.figure(1)
    popularity_likes = df.groupby('topic').agg({'likes': 'sum'}).sort_values(by='likes', ascending=False)
    ax = sns.barplot(x=popularity_likes.index, y='likes', data=popularity_likes)
    plt.title('Популярность тем на основе лайков')
    plt.xlabel('Тема')
    plt.ylabel('Лайки')
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.3)
    plt.figtext(0.5, -0.25, "\n".join(topics), ha="center", fontsize=8, bbox={"boxstyle": "round", "alpha": 0.1})
    plt.savefig('popularity_likes.png', bbox_inches='tight')

    plt.figure(2)
    popularity_views = df.groupby('topic').agg({'views': 'sum'}).sort_values(by='views', ascending=False)
    ax = sns.barplot(x=popularity_views.index, y='views', data=popularity_views)
    plt.title('Популярность тем на основе просмотров')
    plt.xlabel('Тема')
    plt.ylabel('Просмотры')
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.3)
    plt.figtext(0.5, -0.25, "\n".join(topics), ha="center", fontsize=8, bbox={"boxstyle": "round", "alpha": 0.1})
    plt.savefig('popularity_views.png', bbox_inches='tight')

    plt.show()

