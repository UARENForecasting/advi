#!/usr/bin/env python
import logging
import os


from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler
from bokeh.server.server import Server
from bokeh.server.views.static_handler import StaticHandler
from jinja2 import Environment, FileSystemLoader
from tornado.web import RequestHandler, StaticFileHandler


from app.config import MENU_VARS, WS_ORIGIN, PREFIX


env = Environment(loader=FileSystemLoader('app/templates'))


def make_url(arg):
    prefix = PREFIX.lstrip('/')
    if not prefix:
        return f'/{arg}'
    elif arg:
        return f'/{prefix}/{arg}'
    else:
        return f'/{prefix}'


class IndexHandler(RequestHandler):
    def get(self):
        template = env.get_template('app_index.html')
        mv = [(a[0], make_url(a[1])) for a in MENU_VARS]
        self.write(template.render(menu_vars=mv, prefix=PREFIX))


if __name__ == '__main__':
    os.environ['BOKEH_RESOURCES'] = 'cdn'
    logging.basicConfig(level='INFO',
                        format='%(asctime)s %(message)s')
    apps = {make_url(arg): Application(DirectoryHandler(filename='app',
                                                        argv=[arg]))
            for _, arg in MENU_VARS}

    extra_patterns = [(make_url('?'), IndexHandler),
                      (make_url('(favicon.ico)'),
                       StaticFileHandler,
                       {'path': "app/static"}),
                      (make_url('static/(.*)'),
                       StaticHandler)]
    paths = [a[0] for a in extra_patterns]
    paths.extend(list(apps.keys()))
    logging.info('Running on localhost:5006 with paths:\n%s', '\n'.join(paths))
    server = Server(apps,
                    allow_websocket_origin=[WS_ORIGIN],
                    use_xheaders=True,
                    extra_patterns=extra_patterns,
                    use_index=False,
                    redirect_root=False,
                    )
    server.start()
    server.io_loop.start()
