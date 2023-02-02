from flask import Flask, request
app = Flask(__name__)

data = {'key1':"hola", "action":"default"}

def get_data():
    return data

def post_data():
    data["key2"]="hello"
    data["action"]="post"
    return data

def update_data():
    data["key1"]="hi"
    data["action"]="update"
    return data

def delete_data():
    global data
    data = {'key1':"hola", "action":"delete"}
    data["action"]="delete"
    return data

@app.route('/data', methods=['GET', 'POST', 'PUT', 'DELETE'])
def data_route():
    if request.method == 'GET':
        print("records are: ")
        return get_data()
    elif request.method == "POST":
        print("record added: ")
        return post_data()
    elif request.method == "PUT":
        print("record updated: ")
        return update_data()
    elif request.method == "DELETE":
        print("record deleted: ")
        return delete_data()

if __name__ == "__main__":
    app.run(debug=True)