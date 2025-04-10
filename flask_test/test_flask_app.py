from flask import Flask, request, jsonify, render_template  # type: ignore

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("test_index.html")


# Dummy function to simulate getting car details
def get_car_details(departure, destination):
    # This is where you'd implement the logic to get the car details.
    # For this example, we'll just return some hardcoded data.
    return {
        "type_of_car": "UberX",
        "price": "10-13",
        "gps_coordinates": {"lat": 40.7128, "long": -74.0060},
    }


@app.route("/get_car", methods=["GET"])
def get_car():
    departure_address = request.args.get("departure_address")
    destination_address = request.args.get("destination_address")

    car_details = get_car_details(departure_address, destination_address)
    return jsonify(car_details)


if __name__ == "__main__":
    app.run(debug=True)
