let SHELLY_ID = undefined;
let SECOND = 1000;

print("Script started.");
print("MQTT connected:", MQTT.isConnected());

Shelly.call("Mqtt.GetConfig", "", function (res, err_code, err_msg, ud) {
  if (err_code !== 0 || !res) {
    print("Mqtt.GetConfig error:", err_msg);
    return;
  }
  SHELLY_ID = res["topic_prefix"];
  print("SHELLY_ID set:", SHELLY_ID);
});

function timerHandler(user_data) {
  if (!MQTT.isConnected()) {
    print("MQTT not connected, skipping.");
    return;
  }
  if (!SHELLY_ID) {
    print("SHELLY_ID not ready, skipping.");
    return;
  }

  Shelly.call("Switch.GetStatus", { id: 0 }, function (res, err_code, err_msg) {
    if (err_code !== 0 || !res) {
      print("Switch.GetStatus error:", err_msg);
      return;
    }

    let topic = "tele/shelly/" + Shelly.getDeviceInfo().id; 

    let data = {
      tsn: parseInt(Date.now()),
      result: {
        on:         res.output === true,
        label: "Airfryer",
        power_w:    res.apower  !== undefined ? parseFloat(res.apower)  : null,
        voltage_v:  res.voltage !== undefined ? parseFloat(res.voltage) : null,
        current_a:  res.current !== undefined ? parseFloat(res.current) : null,
        energy_wh:  res.aenergy && res.aenergy.total !== undefined ? parseFloat(res.aenergy.total) : null
      }
    };

    let ok = MQTT.publish(topic, JSON.stringify(data), 0, false);
    if (ok === false) {
      print("MQTT.publish failed. Topic:", topic);
    } else {
      print("Message sent to", topic, "at", data.tsn);
    }
  });
}

Timer.set(SECOND, true, timerHandler, null);