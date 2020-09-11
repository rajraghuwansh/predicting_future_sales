from flask import Flask
app = Flask(__name__)

@app.route('/admin')
def hello_admin():
  return 'Hello ADMIN !'
@app.route('/<name>')
def hello_user(name):
  return ' Hello user- %s !' %name

@app.route('/user/<person>')
def which_person(person):
    if person =='admin':
      return redirect(url_for('hello_admin'))
    else :
      return redirect(url_for('hello_user',name=person))
                    
  
if  __name__ == '__main__':
  app.run(debug=True)
                      
                     
                
                      
  
  
