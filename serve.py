import logging
import os


from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler
from bokeh.server.server import Server
from jinja2 import Environment, FileSystemLoader
from tornado.web import RequestHandler, StaticFileHandler


from app.config import MENU_VARS, WS_ORIGIN


env = Environment(loader=FileSystemLoader('app/templates'))


class IndexHandler(RequestHandler):
    def get(self):
        template = env.get_template('app_index.html')
        self.write(template.render(menu_vars=MENU_VARS))


if __name__ == '__main__':
    os.environ['BOKEH_RESOURCES'] = 'cdn'
    logging.basicConfig(level='INFO',
                        format='%(asctime)s %(message)s')
    apps = {f'/{arg}': Application(DirectoryHandler(filename='app',
                                                    argv=[arg]))
            for _, arg in MENU_VARS}
    server = Server(apps,
                    allow_websocket_origin=[WS_ORIGIN],
                    use_xheaders=True,
                    extra_patterns=[('/', IndexHandler),
                                    ('/(favicon.ico)',
                                     StaticFileHandler,
                                     {'path': "app/static"})],
                    use_index=False,
                    redirect_root=False,
                    )
    server.start()
    server.io_loop.start()
