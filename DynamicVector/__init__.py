from .DynamicVector import DynamicVector, __doc__
from ._metadata import (
    __version__,
    __author__,
    __email__,
    __license__,
    __status__,
    __maintainer__,
    __credits__,
    __url__,
    __description__,
    __copyright__,
)

__all__ = [
    "DynamicVector",
]


if __name__ == "__main__":
    _ = __doc__
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Email: {__email__}")
    print(f"License: {__license__}")
    print(f"Status: {__status__}")
    print(f"Maintainer: {__maintainer__}")
    print(f"Credits: {__credits__}")
    print(f"URL: {__url__}")
    print(f"Description: {__description__}")
    print(f"Copyright: {__copyright__}")
