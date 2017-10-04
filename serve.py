#!/usr/bin/env python
import importlib
import logging
import os


from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler
from bokeh.model import Model
from bokeh.server.server import Server
from bokeh.server.views.static_handler import StaticHandler
from bokeh.util.compiler import bundle_models
from jinja2 import Environment, FileSystemLoader
from tornado.web import RequestHandler, StaticFileHandler


from app.config import (MENU_VARS, WS_ORIGIN, PREFIX, GA_TRACKING_ID,
                        CUSTOM_BOKEH_MODELS)


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
        self.write(template.render(menu_vars=mv, prefix=PREFIX,
                                   ga_tracking_id=GA_TRACKING_ID))


def compile_custom_models():
    """
    Pre-compile the custom models so they don't need to be
    recompiled on each page load
    """
    models = []
    for modspec, cm in CUSTOM_BOKEH_MODELS:
        mod = importlib.import_module(modspec)
        custom_model = getattr(mod, cm)
        custom_model.__implementation__ = custom_model.__js_implementation__
        models.append(custom_model)
    js_code = bundle_models(models)

    # must remove the model from the bokeh model map_fig
    # since it has no __implementation__ and will be added
    # in the app
    model_map = Model.model_class_reverse_map
    for _, cm in CUSTOM_BOKEH_MODELS:
        if cm in model_map:
            del model_map[cm]
    return js_code


if __name__ == '__main__':
    os.environ['BOKEH_RESOURCES'] = 'cdn'
    logging.basicConfig(level='INFO',
                        format='%(asctime)s %(message)s')
    custom_js = compile_custom_models()
    apps = {make_url(arg): Application(DirectoryHandler(filename='app',
                                                        argv=[arg, custom_js]))
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
