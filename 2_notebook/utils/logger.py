# -*- coding:utf-8 -*-
from logging import DEBUG, Formatter, StreamHandler, getLogger, handlers


def root_logger():
    # root loggerを取得
    logger = getLogger()

    # formatterを作成
    formatter = Formatter('%(asctime)s %(name)s %(funcName)s [%(levelname)s]: %(message)s')

    # handlerを作成しフォーマッターを設定
    handler = StreamHandler()
    handler.setFormatter(formatter)

    # loggerにhandlerを設定、イベント捕捉のためのレベルを設定
    logger.addHandler(handler)
    # log levelを設定
    logger.setLevel(DEBUG)

    return logger
