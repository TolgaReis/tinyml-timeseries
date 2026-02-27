let SHELLY_ID = undefined;
let SECOND = 1000;

print(MQTT.isConnected());

Shelly.call("Mqtt.GetConfig", "", function (res, err_code, err_msg, ud) {
    SHELLY_ID = res["topic_prefix"];
});

function timerHandler(user_data) {
    let emData = Shelly.getComponentStatus("em", 0);

    let phases = ['a', 'b', 'c'];
    let fields = ['current', 'voltage', 'act_power', 'aprt_power', 'pf'];

    let result = {};
    phases.forEach(function(phase) {
        fields.forEach(function(field) {
            let key = phase + "_" + field;
            result[key] = parseFloat(emData[key]);
        });
    });

    let data = {
        tsn: parseInt(Date.now()),
        result: result
    };

    let topic = "tele/shelly/central_em/" + Shelly.getDeviceInfo().id;

    MQTT.publish(topic, JSON.stringify(data), 0, false); 
    print("Message sent to " + topic + " topic at " + data.tsn);
}

Timer.set(SECOND, true, timerHandler, null);