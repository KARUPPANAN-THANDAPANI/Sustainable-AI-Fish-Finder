from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return """
    <html>
        <body style="background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 50px; text-align: center;">
            <h1>🎣 AI Fish Finder Test</h1>
            <p>If you can see this, Flask is working!</p>
            <p>Server is running correctly.</p>
        </body>
    </html>
    """

if __name__ == '__main__':
    print("🚀 Starting TEST server...")
    print("🌐 Open: http://localhost:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)