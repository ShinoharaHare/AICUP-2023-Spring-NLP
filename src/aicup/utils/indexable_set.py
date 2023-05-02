from typing import Any, Iterable, Optional, Set, Union, Generic, TypeVar, MutableSet

_T = TypeVar('_T')

class IndexableSet(MutableSet[_T], Generic[_T]):
    def __init__(self, iterable: Optional[Iterable[_T]] = None) -> None:
        self._k2v = []
        self._v2k = {}
        
        if iterable:
            for x in iterable:
                self.add(x)

    def add(self, value: _T) -> None:
        if value in self._v2k:
            return
            
        self._v2k[value] = len(self._k2v)
        self._k2v.append(value)

    def discard(self, value: _T) -> None:
        if value in self._v2k:
            i = self._v2k[value]
            self._k2v[i], self._k2v[-1] = self._k2v[-1], self._k2v[i]
            self._v2k[self._k2v[i]] = i
            self._v2k.pop(self._k2v.pop())

    def __contains__(self, x: object) -> bool:
        return x in self._v2k

    def __len__(self):
        return len(self._k2v)

    def __getitem__(self, i: int):
        return self._k2v[i]
    
    def __iter__(self):
        return iter(self._k2v)
    
    # def __repr__(self):
    #     return f'IndexableSet({", ".join([str(v) for v in self._k2v.values()])})'
    
