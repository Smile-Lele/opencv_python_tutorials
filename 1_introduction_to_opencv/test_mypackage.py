from mypackage import IFile

ifile = IFile('../mydata/1.json')
print(ifile.read())

data = {1:3}
ifile.write(data)

print(ifile.read())
