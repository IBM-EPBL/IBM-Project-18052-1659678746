import { Vonage } from "@vonage/server-sdk";

var alert_message = document.getElementById("alert-message")

var activate_alarm = document.getElementById("activateAlarm-btn")

        activate_alarm.addEventListener("click", () => {
            // window.alert("alarm button pressed!")
            console.log(alert_message.value)
        })


const vonage = new Vonage({
    apiKey: '952256f8',
    apiSecret: 'JW63QuQ2H40INwdn'
})

const from = "Vonage APIs"
const to = "918610674093"
const text = alert_message

vonage.message.sendSMs(from, to, text, (err, responseData) => {
    if (err) {
        console.log(err);
    } else {
        if(responseData.messages[0]['status'] === "0") {
            console.log("Message sent successfully.");
        } else {
            console.log(`Message failed with error: ${responseData.messages[0]['error-text']}`);
        }
    }
})

