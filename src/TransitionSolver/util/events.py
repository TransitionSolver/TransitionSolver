from typing import Any, Callable


class NotifyHandler:
    events: dict = {}

    def handleEvent(self, caller: Any, eventName: str):
        fullName = type(caller).__name__ + '-' + eventName
        if fullName in self.events:
            self.events[fullName](caller)

    def addEvent(self, name: str, reaction: Callable[[Any], None]):
        self.events[name] = reaction

# Create a global singleton when this module is imported.
if not 'notifyHandler' in globals():
    notifyHandler = NotifyHandler()
