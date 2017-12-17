import tweepy
import os
import datetime as dt


def get_tweets(topic, cnt=30):
    consumer_key='7gkx93O6yJxDgX4CblWWvJaPt'
    consumer_secret='AdympVlMXibTyspMIbWjB5bBSBKgmdWh3CqBBs2cmYk2SugErc'
    access_token_key='780472913279475713-n8FH0ENAo18F5eXOCQ6UU1jiYoYaxM3'
    access_token_secret='SnTjRo7h8ycvRISDS9dzGto6aX7dgK3GH1wRF3OMD2BbF'


    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token_key,access_token_secret)

    api = tweepy.API(auth)

    public_tweets = api.search(topic, count=cnt)

    currdir = os.getcwd()
    newpath = currdir + '/store/%s' %topic
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir('%s' % newpath)

    i=0
    tw = []
    date = dt.datetime.now().strftime("%Y%m%d%H%M%S")

    for tweet in public_tweets:
        ff = open('./%s_%s.txt' %(date, i), 'w')
        i += 1
        tw.append(tweet.text)
        ff.write(tweet.text)

    os.chdir('%s' % currdir)
    j =0

    for tweet in public_tweets:
        ff = open('./temp/%s.txt' % j, 'w')
        j += 1
        ff.write(tweet.text)
    #os.chdir('%s' % currdir)

    return tw
