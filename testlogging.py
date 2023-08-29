import logging
logging.basicConfig(level=logging.INFO, filename='sample.log', encoding='utf-8')
logger = logging.getLogger(__name__)
logger.info('Hello World')