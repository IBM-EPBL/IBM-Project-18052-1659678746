var alarm = document.getElementById("activateAlarm-btn")
var audio = new Audio("alarm.mp3")

alarm.addEventListener("click", ()=>{
    window.alert("alarm button pressed!")
    audio.play()
})