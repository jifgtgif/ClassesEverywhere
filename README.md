# ClassesEverywhere
official repo for ClassesEverywhere programming language

features:
- very OOP!
- everything is defined as an event OR class
- classes are derived from Thing (empty)
```ce
Thing/SomethingElse = {
  int property = 0;
  function addProperty(delta) {
    property=property+delta;
  }
}

on Events.start {
  SomethingElseElse = new("Thing/SomethingElse");
  print(SomethingElseElse.property);
  SomethingElseElse.addProperty(1);
  print(SomethingElseElse.property);
  SomethingElseElse.addProperty(3);
  print(SomethingElseElse.property);
}
```
would output 
```
0
1
4
```
in fact... running 
```
py main.py file.ce
```
would output just that!
