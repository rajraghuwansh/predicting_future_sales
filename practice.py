from flask import Flask render temlate ,request
app = Flask(__name__)
@app.route( '/')
def hell_world():
  return 'hello world'
if __name__ == '__main__':
  app.run(debug= True)
           
