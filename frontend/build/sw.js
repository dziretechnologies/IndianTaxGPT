const variable = "appV1"

this.addEventListener("install",(event)=>[
    event.waitUntil(
        caches.open(variable).then((cache)=>{
            cache.addAll(['/*'])
        })
    )
])