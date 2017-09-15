import logging


from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler
from bokeh.server.server import Server
from jinja2 import Environment, FileSystemLoader
from tornado.web import RequestHandler


env = Environment(loader=FileSystemLoader('templates'))


MENU_VARS = (('2m Temperature', 'temp'),
             ('1 hr Temperature Change', 'dt'),
             ('10m Wind Speed', 'wspd'),
             ('1 hr Precip', 'rain'),
             ('Accumulated Precip', 'rainac'),
             ('GHI', 'ghi'),
             ('DNI', 'dni'))


class IndexHandler(RequestHandler):
    def get(self):
        template = env.get_template('app_index.html')
        self.write(template.render(menu_vars=MENU_VARS))


apps = {f'/{fname}': Application(DirectoryHandler(filename=fname))
        for _, fname in MENU_VARS}
server = Server(apps,
                extra_patterns=[('/', IndexHandler)])
server.start()

if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    server.io_loop.start()
